from .sessions import Session, CaptionSegment


def _get_text(seg: CaptionSegment, lang: str) -> str:
    if lang == "original":
        return seg.original_text
    return seg.translations.get(lang, "")


def _fmt_ts_srt(t: float) -> str:
    t = max(t, 0)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_ts_vtt(t: float) -> str:
    t = max(t, 0)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def export_srt(session: Session, lang: str) -> str:
    lines = []
    base = session.segments[0].start_ts if session.segments else 0
    i = 0
    for seg in session.segments:
        text = _get_text(seg, lang)
        if not text:
            continue
        i += 1
        start = seg.start_ts - base
        end = max(seg.end_ts - base, start + 0.5)
        lines.append(str(i))
        lines.append(f"{_fmt_ts_srt(start)} --> {_fmt_ts_srt(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def export_vtt(session: Session, lang: str) -> str:
    lines = ["WEBVTT", ""]
    base = session.segments[0].start_ts if session.segments else 0
    for seg in session.segments:
        text = _get_text(seg, lang)
        if not text:
            continue
        start = seg.start_ts - base
        end = max(seg.end_ts - base, start + 0.5)
        lines.append(f"{_fmt_ts_vtt(start)} --> {_fmt_ts_vtt(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def export_txt(session: Session, lang: str) -> str:
    return "\n".join(_get_text(seg, lang) for seg in session.segments if _get_text(seg, lang))
