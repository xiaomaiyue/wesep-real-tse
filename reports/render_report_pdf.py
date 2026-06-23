#!/usr/bin/env python3
"""Fallback PDF renderer for the LaTeX report.

The workspace currently has no xelatex/pdflatex. This script renders a readable
PDF from the LaTeX source using macOS AppKit, keeping the .tex file as the
authoritative source.
"""
import re
import sys
from pathlib import Path

from AppKit import (NSAttributedString, NSColor, NSFont, NSMakeRect,
                    NSMakePoint, NSRectFill,
                    NSPrintJobDisposition, NSPrintJobSavingURL,
                    NSMutableAttributedString, NSPortraitOrientation,
                    NSPrintOperation, NSPrintInfo, NSPrintSaveJob, NSView)
from Foundation import NSDictionary, NSMakeSize, NSURL, NSString


def strip_latex(src):
    m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", src, re.S)
    text = m.group(1) if m else src
    drops = [
        r"\\maketitle", r"\\tableofcontents", r"\\newpage",
        r"\\centering", r"\\toprule", r"\\midrule", r"\\bottomrule",
        r"\\hline",
    ]
    for pat in drops:
        text = re.sub(pat, "", text)

    text = re.sub(r"\\section\{([^{}]+)\}", r"\n\n# \1\n", text)
    text = re.sub(r"\\subsection\{([^{}]+)\}", r"\n\n## \1\n", text)
    text = re.sub(r"\\subsubsection\{([^{}]+)\}", r"\n\n### \1\n", text)
    text = re.sub(r"\\caption\{([^{}]+)\}", r"\n[表注] \1\n", text)

    text = re.sub(r"\\begin\{(?:itemize|enumerate|quote|center|table|longtable|tabularx|tabular)\}(?:\[[^\]]*\]|\{[^{}]*\})*", "\n", text)
    text = re.sub(r"\\end\{(?:itemize|enumerate|quote|center|table|longtable|tabularx|tabular)\}", "\n", text)
    text = re.sub(r"\\begin\{minipage\}\{[^{}]*\}", "\n", text)
    text = re.sub(r"\\end\{minipage\}", "\n", text)
    text = re.sub(r"\\fcolorbox\{[^{}]*\}\{[^{}]*\}\{", "", text)

    text = re.sub(r"\\item\s*", "  - ", text)

    def one_arg(cmd, s):
        return re.sub(r"\\" + cmd + r"\{([^{}]*)\}", r"\1", s)

    for cmd in ["textbf", "textit", "metric", "good", "warn", "emph"]:
        text = one_arg(cmd, text)
    text = re.sub(r"\\code\{([^{}]*)\}", r"`\1`", text)
    text = re.sub(r"\\detokenize\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\texttt\{([^{}]*)\}", r"`\1`", text)
    text = re.sub(r"\\textcolor\{[^{}]*\}\{([^{}]*)\}", r"\1", text)

    text = text.replace(r"\%", "%")
    text = text.replace(r"\_", "_")
    text = text.replace(r"\&", "&")
    text = text.replace(r"\rightarrow", "->")
    text = text.replace(r"\approx", "≈")
    text = text.replace(r"\sim", "~")
    text = text.replace(r"\lambda", "lambda")
    text = text.replace(r"\cos", "cos")

    # Make LaTeX table rows readable as aligned text.
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        if re.fullmatch(r"[pXlcr|0-9.]+", line):
            continue
        if "&" in line and r"\\" in line:
            line = line.replace(r"\\", "")
            cells = [c.strip() for c in line.split("&")]
            line = " | ".join(cells)
        else:
            line = line.replace(r"\\", "")
        lines.append(line)
    text = "\n".join(lines)

    # Remove simple math delimiters and remaining LaTeX noise.
    text = re.sub(r"\\\[(.*?)\\\]", lambda m: "\n" + m.group(1).strip() + "\n", text, flags=re.S)
    text = text.replace("$", "")
    text = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^{}]*\})?", "", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def attr_text(plain):
    body_font = NSFont.fontWithName_size_("Noto Serif CJK SC", 10.5) or NSFont.systemFontOfSize_(10.5)
    mono_font = NSFont.fontWithName_size_("Menlo", 9.0) or NSFont.userFixedPitchFontOfSize_(9.0)
    h1_font = NSFont.boldSystemFontOfSize_(18)
    h2_font = NSFont.boldSystemFontOfSize_(14)
    attrs_body = NSDictionary.dictionaryWithObjectsAndKeys_(body_font, "NSFont", NSColor.blackColor(), "NSColor")
    attrs_mono = NSDictionary.dictionaryWithObjectsAndKeys_(mono_font, "NSFont", NSColor.blackColor(), "NSColor")
    attrs_h1 = NSDictionary.dictionaryWithObjectsAndKeys_(h1_font, "NSFont", NSColor.colorWithCalibratedRed_green_blue_alpha_(0.08, 0.22, 0.40, 1), "NSColor")
    attrs_h2 = NSDictionary.dictionaryWithObjectsAndKeys_(h2_font, "NSFont", NSColor.colorWithCalibratedRed_green_blue_alpha_(0.08, 0.22, 0.40, 1), "NSColor")

    out = NSMutableAttributedString.alloc().init()
    for line in plain.splitlines(True):
        stripped = line.strip()
        attrs = attrs_body
        content = line
        if stripped.startswith("# "):
            attrs = attrs_h1
            content = stripped[2:] + "\n"
        elif stripped.startswith("## "):
            attrs = attrs_h2
            content = stripped[3:] + "\n"
        elif "|" in line or stripped.startswith("`") or stripped.startswith("  - `"):
            attrs = attrs_mono
        out.appendAttributedString_(NSAttributedString.alloc().initWithString_attributes_(content, attrs))
    return out


class ReportView(NSView):
    def initWithAttributedString_frame_(self, attributed, frame):
        self = self.initWithFrame_(frame)
        self.attributed = attributed
        return self

    def knowsPageRange_(self, range_out):
        self.page_width = self.bounds().size.width
        self.page_height = self.bounds().size.height
        self.margin = 54
        content_width = self.page_width - 2 * self.margin
        content_height = self.page_height - 2 * self.margin
        self.frames = []
        text = str(self.attributed.string())
        chars_per_line = max(int(content_width / 6.0), 40)
        lines_per_page = max(int(content_height / 15.0), 30)
        lines = []
        for para in text.splitlines():
            if not para:
                lines.append("")
                continue
            while len(para) > chars_per_line:
                lines.append(para[:chars_per_line])
                para = para[chars_per_line:]
            lines.append(para)
        for i in range(0, len(lines), lines_per_page):
            self.frames.append("\n".join(lines[i:i + lines_per_page]))
        range_out.location = 1
        range_out.length = max(len(self.frames), 1)
        return True

    def rectForPage_(self, page):
        return NSMakeRect(0, 0, self.page_width, self.page_height)

    def drawRect_(self, rect):
        page = self.currentPage() - 1
        if page < 0 or page >= len(self.frames):
            return
        chunk = self.frames[page]
        attributed = attr_text(chunk)
        draw_rect = NSMakeRect(self.margin, self.margin, self.page_width - 2 * self.margin, self.page_height - 2 * self.margin)
        attributed.drawInRect_(draw_rect)


def render(tex_path, pdf_path):
    plain = strip_latex(Path(tex_path).read_text(encoding="utf-8"))
    render_plain_long_pdf(plain, pdf_path)


class PlainLongView(NSView):
    def initWithLines_frame_(self, lines, frame):
        self = self.initWithFrame_(frame)
        self.lines = lines
        self.body_font = NSFont.fontWithName_size_("Noto Serif CJK SC", 10.5) or NSFont.systemFontOfSize_(10.5)
        self.mono_font = NSFont.fontWithName_size_("Menlo", 9.0) or NSFont.userFixedPitchFontOfSize_(9.0)
        self.h1_font = NSFont.boldSystemFontOfSize_(18)
        self.h2_font = NSFont.boldSystemFontOfSize_(14)
        return self

    def drawRect_(self, rect):
        NSColor.whiteColor().set()
        NSRectFill(self.bounds())
        y = self.bounds().size.height - 54
        x = 54
        for level, line in self.lines:
            if level == 1:
                font = self.h1_font
                color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.08, 0.22, 0.40, 1)
                dy = 25
            elif level == 2:
                font = self.h2_font
                color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.08, 0.22, 0.40, 1)
                dy = 20
            elif level == 3:
                font = self.mono_font
                color = NSColor.blackColor()
                dy = 15
            else:
                font = self.body_font
                color = NSColor.blackColor()
                dy = 15
            attrs = NSDictionary.dictionaryWithObjectsAndKeys_(font, "NSFont", color, "NSColor")
            NSString.stringWithString_(line).drawAtPoint_withAttributes_(NSMakePoint(x, y), attrs)
            y -= dy


def wrap_plain(plain, width_chars=82):
    wrapped = []
    for raw in plain.splitlines():
        level = 0
        line = raw.rstrip()
        if line.startswith("# "):
            level = 1
            line = line[2:]
        elif line.startswith("## "):
            level = 2
            line = line[3:]
        elif "|" in line or line.strip().startswith("`"):
            level = 3
        if not line:
            wrapped.append((0, ""))
            continue
        limit = 70 if level in (1, 2) else width_chars
        while len(line) > limit:
            cut = line.rfind(" ", 0, limit)
            if cut < limit * 0.45:
                cut = limit
            wrapped.append((level, line[:cut]))
            line = line[cut:].lstrip()
        wrapped.append((level, line))
    return wrapped


def render_plain_long_pdf(plain, pdf_path):
    width = 595
    lines = wrap_plain(plain)
    height = max(842, 108 + len(lines) * 15)
    view = PlainLongView.alloc().initWithLines_frame_(lines, NSMakeRect(0, 0, width, height))
    data = view.dataWithPDFInsideRect_(NSMakeRect(0, 0, width, height))
    ok = data.writeToFile_atomically_(str(Path(pdf_path).resolve()), True)
    if not ok:
        raise SystemExit("PDF render failed")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: render_report_pdf.py report.tex report.pdf")
    render(sys.argv[1], sys.argv[2])
