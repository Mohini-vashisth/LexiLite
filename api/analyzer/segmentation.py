"""
Splits extracted document text into clause-sized chunks for the classifier.

Why this exists (LEX-17): PDF text arrives hard-wrapped at ~100 characters per
line, with a "Page N" header at the top of every page and blank lines almost
only at page breaks. Splitting on blank lines therefore produced one
"clause" per PDF page (1,400-4,000 chars). The classifier was trained on
single clauses (median 192 chars, p99 ~1,300) and its encoder reads at most
256 tokens (~1,000 chars), so ~70% of every page was never seen by the model.

What this does, in order:
  1. Drop page furniture ("Page 4", bare page numbers, ____ rules). A clause
     that runs across a page break is kept whole.
  2. Group lines into paragraphs: a blank line ends one, and so does the
     start of a numbered/lettered/heading line ("3)", "4.", "B. Insurance",
     "Section 5", "ARTICLE IV", ALL-CAPS headings). Wrapped lines inside a
     paragraph are re-joined into normal sentences.
     Exception: text with no blank lines where nearly every line already ends
     in punctuation is "one clause per line" (a simple paste), so each line is
     kept as its own paragraph.
  3. Merge fragments (very short pieces, headings with no body) into the
     paragraph that follows, so a heading travels with its text.
  4. Split anything longer than `max_chars` at sentence boundaries, packing
     sentences greedily; a single over-long sentence is split at semicolons,
     then at whitespace. Nothing is ever truncated.

`max_chars` defaults to 800, which keeps a chunk safely inside the encoder's
256-token window while staying close to the training distribution.
"""

import re

DEFAULT_MAX_CHARS = 800
MIN_CHARS = 40

# "Page 4", "Page 4 of 12", "- 4 -", or just "4" alone on a line.
_PAGE_FURNITURE = re.compile(r'^(?:page\s+\d+(?:\s+of\s+\d+)?|-\s*\d{1,4}\s*-|\d{1,4})$', re.IGNORECASE)
# ________ or -------- or ======== lines (signature lines, separators).
_RULE_LINE = re.compile(r'^[\s_\-=*.]{3,}$')

# A line that begins a new clause: "3)", "4.", "2.1.", "B.", "IV.", "Section 5",
# "Article IV". Parenthesised sub-items like "(a)" are deliberately NOT starts:
# they belong to their parent clause. (Letter/roman rules are case-sensitive.)
_CLAUSE_START = re.compile(
    r'^(?:'
    r'\d+(?:\.\d+)*[.)]\s'
    r'|[A-Z][.)]\s'
    r'|[IVX]{2,}[.)]\s'
    r'|(?i:section|article|paragraph|clause)\s+[\dIVXivx]+'
    r')'
)

_SENTENCE_END = re.compile(r'[.;:!?]["\')”’]?$')
_SENTENCE_BREAK = re.compile(r'(?<=[.!?])\s+(?=["“(A-Z0-9])')
_FRAGMENT_END = re.compile(r'[.;:!?)"”’]$')


def _is_clause_start(line):
    if _CLAUSE_START.match(line):
        return True
    # ALL-CAPS heading such as "TERMINATION" or "LIMITATION OF LIABILITY".
    letters = [c for c in line if c.isalpha()]
    return (
        len(line) <= 80
        and len(letters) >= 3
        and line.isupper()
        and not re.search(r'[.,;:]$', line)
    )


def _unwrap(lines):
    """Join hard-wrapped lines into one string with single spaces."""
    out = lines[0]
    for line in lines[1:]:
        # "non-" + "discrimination": hyphenated line break, no extra space.
        out += line if out.endswith('-') else ' ' + line
    return re.sub(r'\s+', ' ', out).strip()


def _is_one_clause_per_line(lines):
    return len(lines) >= 2 and sum(1 for l in lines if _SENTENCE_END.search(l)) / len(lines) >= 0.8


def _paragraphs(text):
    raw_lines = [l.strip() for l in text.splitlines()]
    has_blank = any(not l for l in raw_lines)
    content = [l for l in raw_lines if l and not _PAGE_FURNITURE.match(l) and not _RULE_LINE.match(l)]

    if not has_blank and _is_one_clause_per_line(content):
        return content

    paras, cur = [], []

    def flush():
        if cur:
            paras.append(_unwrap(cur))
            cur.clear()

    for line in raw_lines:
        if not line:
            flush()
            continue
        if _PAGE_FURNITURE.match(line) or _RULE_LINE.match(line):
            continue  # dropped, but does NOT end the paragraph (clauses span pages)
        if cur and _is_clause_start(line):
            flush()
        cur.append(line)
    flush()
    return paras


def _is_fragment(p, min_chars):
    return len(p) < min_chars or (len(p) <= 80 and not _FRAGMENT_END.search(p))


def _merge_fragments(paras, min_chars):
    out, carry = [], ''
    for p in paras:
        p = f'{carry} {p}'.strip() if carry else p
        carry = ''
        if _is_fragment(p, min_chars):
            carry = p
            continue
        out.append(p)
    if carry:
        if out:
            out[-1] = f'{out[-1]} {carry}'
        else:
            out.append(carry)
    return out


def _hard_split(sentence, max_chars):
    """Split one over-long sentence: at semicolons if possible, else whitespace."""
    if len(sentence) <= max_chars:
        return [sentence]
    pieces = []
    rest = sentence
    while len(rest) > max_chars:
        cut = rest.rfind('; ', 0, max_chars)
        cut = cut + 1 if cut > 0 else rest.rfind(' ', 0, max_chars)
        if cut <= 0:
            cut = max_chars
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        pieces.append(rest)
    return pieces


def _split_long(paragraph, max_chars, min_chars):
    if len(paragraph) <= max_chars:
        return [paragraph]
    chunks, cur = [], ''
    for sentence in _SENTENCE_BREAK.split(paragraph):
        for piece in _hard_split(sentence, max_chars):
            if cur and len(cur) + 1 + len(piece) > max_chars:
                chunks.append(cur)
                cur = piece
            else:
                cur = f'{cur} {piece}'.strip()
    if cur:
        if chunks and len(cur) < min_chars:
            chunks[-1] = f'{chunks[-1]} {cur}'  # never leave a tiny orphan tail
        else:
            chunks.append(cur)
    return chunks


def segment_clauses(text, max_chars=DEFAULT_MAX_CHARS, min_chars=MIN_CHARS):
    """Return the clause-sized chunks of `text` (all of them, in order)."""
    if not text or not text.strip():
        return []
    chunks = []
    for paragraph in _merge_fragments(_paragraphs(text), min_chars):
        chunks.extend(_split_long(paragraph, max_chars, min_chars))
    return [c for c in chunks if c]
