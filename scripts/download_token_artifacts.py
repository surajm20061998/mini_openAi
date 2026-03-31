from __future__ import annotations

import argparse
import html
from html.parser import HTMLParser
import http.cookiejar
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024
DEFAULT_DATA_DIR = "data"
DEFAULT_FILENAMES = {
    "train": "train_tokens_full_w8.npy",
    "val": "val_tokens_full_w8.npy",
    "vocab": "vocab.json",
    "merges": "merges.json",
}
USER_AGENT = "mini_openAi-gdrive-downloader/1.0"


class _DriveConfirmationParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.forms: list[dict[str, object]] = []
        self.links: list[str] = []
        self._current_form: dict[str, object] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag == "form":
            self._current_form = {
                "action": attr_map.get("action", ""),
                "method": (attr_map.get("method") or "get").lower(),
                "inputs": {},
            }
            return

        if tag == "input" and self._current_form is not None:
            name = attr_map.get("name")
            if name:
                inputs = self._current_form["inputs"]
                assert isinstance(inputs, dict)
                inputs[name] = attr_map.get("value", "")
            return

        if tag == "a":
            href = attr_map.get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._current_form is not None:
            self.forms.append(self._current_form)
            self._current_form = None


def _extract_file_id(source: str) -> str:
    source = source.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", source):
        return source

    patterns = [
        r"/file/d/([A-Za-z0-9_-]+)",
        r"[?&]id=([A-Za-z0-9_-]+)",
        r"/d/([A-Za-z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, source)
        if match:
            return match.group(1)

    raise ValueError(
        f"Could not extract a Google Drive file id from: {source!r}. "
        "Pass either a public share URL or the raw file id."
    )


def _build_initial_download_url(file_id: str) -> str:
    query = urllib.parse.urlencode({"export": "download", "id": file_id})
    return f"https://drive.google.com/uc?{query}"


def _build_opener() -> tuple[urllib.request.OpenerDirector, http.cookiejar.CookieJar]:
    cookies = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies))
    opener.addheaders = [("User-Agent", USER_AGENT)]
    return opener, cookies


def _is_probably_html(response: urllib.response.addinfourl, first_chunk: bytes) -> bool:
    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" in content_type:
        return True

    sample = first_chunk.lstrip().lower()
    return sample.startswith(b"<!doctype html") or sample.startswith(b"<html")


def _content_disposition_filename(response: urllib.response.addinfourl) -> str | None:
    disposition = response.headers.get("Content-Disposition", "")
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', disposition)
    if not match:
        return None
    return urllib.parse.unquote(match.group(1))


def _cookie_confirm_url(
    cookies: http.cookiejar.CookieJar,
    file_id: str,
) -> str | None:
    for cookie in cookies:
        if cookie.name.startswith("download_warning"):
            query = urllib.parse.urlencode(
                {
                    "export": "download",
                    "confirm": cookie.value,
                    "id": file_id,
                }
            )
            return f"https://drive.google.com/uc?{query}"
    return None


def _page_confirm_url(page_text: str, base_url: str) -> str | None:
    parser = _DriveConfirmationParser()
    parser.feed(page_text)

    for form in parser.forms:
        action = str(form.get("action") or "")
        inputs = form.get("inputs")
        if not action or not isinstance(inputs, dict):
            continue
        if not any(key in inputs for key in ("id", "confirm", "uuid", "export")):
            continue

        query = urllib.parse.urlencode(
            {
                str(key): str(value)
                for key, value in inputs.items()
                if value is not None
            }
        )
        resolved_action = urllib.parse.urljoin(base_url, html.unescape(action))
        separator = "&" if urllib.parse.urlparse(resolved_action).query else "?"
        return f"{resolved_action}{separator}{query}" if query else resolved_action

    for link in parser.links:
        if not any(token in link for token in ("confirm=", "export=download", "/download")):
            continue
        return urllib.parse.urljoin(base_url, html.unescape(link))

    return None


def _open_resolved_download(
    opener: urllib.request.OpenerDirector,
    cookies: http.cookiejar.CookieJar,
    file_id: str,
) -> tuple[urllib.response.addinfourl, bytes]:
    url = _build_initial_download_url(file_id)
    last_page_text = ""

    for _ in range(5):
        response = opener.open(url)
        first_chunk = response.read(64 * 1024)

        if not _is_probably_html(response, first_chunk):
            return response, first_chunk

        page_text = (first_chunk + response.read()).decode("utf-8", errors="ignore")
        response.close()
        last_page_text = page_text

        confirm_url = _cookie_confirm_url(cookies, file_id) or _page_confirm_url(
            page_text=page_text,
            base_url=url,
        )
        if confirm_url is None:
            break
        url = confirm_url

    if "Google Drive" in last_page_text:
        raise RuntimeError(
            "Google Drive did not return a downloadable file. "
            "Make sure the file is shared as 'Anyone with the link' and that the URL/file id is correct."
        )
    raise RuntimeError("Could not resolve a direct download URL from the provided Google Drive link.")


def _format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def _stream_to_path(
    response: urllib.response.addinfourl,
    initial_chunk: bytes,
    output_path: Path,
    chunk_size: int,
) -> None:
    total_size_header = response.headers.get("Content-Length")
    total_size = int(total_size_header) if total_size_header and total_size_header.isdigit() else None
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    downloaded = 0

    try:
        with tmp_path.open("wb") as handle:
            if initial_chunk:
                handle.write(initial_chunk)
                downloaded += len(initial_chunk)
                _print_progress(output_path.name, downloaded, total_size)

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                _print_progress(output_path.name, downloaded, total_size)

        tmp_path.replace(output_path)
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _print_progress(name: str, downloaded: int, total_size: int | None) -> None:
    if total_size and total_size > 0:
        pct = min(100.0, 100.0 * downloaded / total_size)
        message = (
            f"\r[download] {name} | {pct:5.1f}% "
            f"| {_format_size(downloaded)}/{_format_size(total_size)}"
        )
    else:
        message = f"\r[download] {name} | {_format_size(downloaded)}"
    sys.stdout.write(message)
    sys.stdout.flush()


def _download_item(source: str, output_path: Path, chunk_size: int, force: bool) -> None:
    if output_path.exists() and not force:
        print(f"[skip] {output_path} already exists")
        return

    file_id = _extract_file_id(source)
    opener, cookies = _build_opener()

    print(f"[start] downloading {output_path.name} -> {output_path}")
    try:
        response, first_chunk = _open_resolved_download(opener, cookies, file_id)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"HTTP error while downloading {output_path.name}: {exc.code} {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error while downloading {output_path.name}: {exc.reason}") from exc

    inferred_name = _content_disposition_filename(response)
    if inferred_name and inferred_name != output_path.name:
        print(f"[info] remote filename={inferred_name}")

    try:
        _stream_to_path(
            response=response,
            initial_chunk=first_chunk,
            output_path=output_path,
            chunk_size=chunk_size,
        )
    finally:
        response.close()

    print(f"[done] saved {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download token artifacts from public Google Drive links into the local data directory."
    )
    p.add_argument("--train", type=str, required=True, help="Google Drive URL or file id for train_tokens_full_w8.npy")
    p.add_argument("--val", type=str, required=True, help="Google Drive URL or file id for val_tokens_full_w8.npy")
    p.add_argument("--vocab", type=str, default=None, help="Optional Google Drive URL or file id for vocab.json")
    p.add_argument("--merges", type=str, default=None, help="Optional Google Drive URL or file id for merges.json")
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory where downloaded files will be stored")
    p.add_argument("--train_name", type=str, default=DEFAULT_FILENAMES["train"])
    p.add_argument("--val_name", type=str, default=DEFAULT_FILENAMES["val"])
    p.add_argument("--vocab_name", type=str, default=DEFAULT_FILENAMES["vocab"])
    p.add_argument("--merges_name", type=str, default=DEFAULT_FILENAMES["merges"])
    p.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Streaming chunk size in bytes")
    p.add_argument("--force", action="store_true", help="Overwrite files if they already exist")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    downloads: list[tuple[str, Path]] = [
        (args.train, data_dir / args.train_name),
        (args.val, data_dir / args.val_name),
    ]
    if args.vocab:
        downloads.append((args.vocab, data_dir / args.vocab_name))
    if args.merges:
        downloads.append((args.merges, data_dir / args.merges_name))

    print(f"[setup] data_dir={data_dir.resolve()}")
    for source, output_path in downloads:
        _download_item(
            source=source,
            output_path=output_path,
            chunk_size=args.chunk_size,
            force=args.force,
        )


if __name__ == "__main__":
    main()
