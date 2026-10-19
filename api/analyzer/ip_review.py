"""
IP-assignment review: the one contract type LexiLite checks.

This is deliberately not the general risky/safe classifier. A clause is only
reviewed if it is about intellectual property; every other clause is reported
as "not checked", never as "safe". Each issue comes from a curated library, so
the reason, the sentence to paste into an email and the "get a lawyer when"
note are fixed, reviewable text rather than free model output.

Every library entry is a DRAFT until a lawyer has reviewed it. The status is
carried into every result so the page can say so.
"""
import re

REVIEW_STATUS = 'draft_pending_lawyer_review'

CHECKED_SCOPE = 'IP assignment clauses (ownership of work, pre-existing materials, licences, moral rights)'

NOT_CHECKED = [
    'Payment terms',
    'Liability and indemnity',
    'Termination',
    'Confidentiality',
    'Non-compete and non-solicitation',
    'Governing law and dispute resolution',
    'Any other clause that is not about intellectual property',
]

# A clause is in scope only if it mentions something IP-related.
IP_TOPIC = re.compile(
    r'intellectual property|copyright|work product|ownership|assign|'
    r'licen[cs]e|moral rights|work made for hire|work for hire|pre-?existing|materials',
    re.IGNORECASE,
)

# Each issue: a pattern that triggers it, and the text a freelancer sees.
ISSUE_LIBRARY = [
    {
        'id': 'all_work_assigned',
        'pattern': re.compile(
            r'\ball\b.{0,40}\b(work|work product|deliverables?|materials|results|output)\b.{0,80}'
            r'\b(vest|vests|assign|assigns|transfer|transfers|belong|belongs)\b',
            re.IGNORECASE | re.DOTALL,
        ),
        'reason': 'Ownership of everything you create can pass to the client, including work outside this project.',
        'paste_sentence': (
            'Could we limit the assignment to the deliverables created specifically for this project, '
            'and keep ownership of anything created outside it?'
        ),
        'lawyer_when': 'The assignment covers work outside this project, or the fee is large.',
    },
    {
        'id': 'pre_existing_materials',
        'pattern': re.compile(r'pre-?existing|existing materials|background (ip|intellectual property)', re.IGNORECASE),
        'reason': 'Your existing templates, tools or portfolio pieces may be handed over with the deliverables.',
        'paste_sentence': (
            'We would like to keep ownership of our pre-existing materials and give you a licence to use '
            'them as part of the deliverables.'
        ),
        'lawyer_when': 'Your templates or tools are central to how you earn a living.',
    },
    {
        'id': 'ownership_before_payment',
        'pattern': re.compile(
            r'\b(upon creation|on creation|immediately|as (they|it) (are|is) created)\b',
            re.IGNORECASE,
        ),
        'reason': 'Ownership can pass to the client as soon as the work is created, before you are paid.',
        'paste_sentence': (
            'Could ownership transfer only once the invoice for those deliverables has been paid in full?'
        ),
        'lawyer_when': 'The unpaid amount is large, or the client has paid late before.',
        'requires_no_payment_condition': True,
    },
    {
        'id': 'perpetual_or_worldwide',
        'pattern': re.compile(r'\b(perpetual|irrevocable|worldwide|in all media|any medium)\b', re.IGNORECASE),
        'reason': 'The rights have no time or territory limit and may cover uses you never priced.',
        'paste_sentence': (
            'Could the rights be limited to this project and an agreed period, with any further use priced separately?'
        ),
        'lawyer_when': 'The fee is large, or the client is a well-known brand that may reuse the work widely.',
    },
    {
        'id': 'moral_rights_waiver',
        'pattern': re.compile(r'moral rights|\bwaive', re.IGNORECASE),
        'reason': (
            'You may be asked to give up moral rights, which cover credit and objecting to changes to your work.'
        ),
        'paste_sentence': (
            'Could we keep the right to be credited for our work and to object to changes that distort it?'
        ),
        'lawyer_when': 'Credit matters for your portfolio, or the client plans major changes to the work.',
    },
    {
        'id': 'work_for_hire_wording',
        'pattern': re.compile(r'work made for hire|work for hire', re.IGNORECASE),
        'reason': (
            'This wording can be read as giving the client ownership by default. Whether it applies depends '
            'on the law, so check it with a lawyer.'
        ),
        'paste_sentence': (
            'Could we replace the work-for-hire wording with an assignment that takes effect on payment?'
        ),
        'lawyer_when': 'Always, before you sign.',
    },
]


def _payment_condition_present(text):
    return re.search(r'\b(paid|payment|invoice|fees?)\b', text, re.IGNORECASE) is not None


def review_ip_clauses(clauses):
    """
    Review a list of clause strings for IP-assignment issues.

    Returns a dict the API and the page both use:
      - checked_scope / not_checked: what this review covers, stated every time
      - status: REVIEW_STATUS, so the library's draft state is never hidden
      - unreadable: True when there is no readable text, so "no issues" is never
        mistaken for a clean contract
      - ip_clause_count: how many clauses were in scope
      - flags: one entry per in-scope clause that matched at least one issue
    """
    texts = [c for c in (clauses or []) if isinstance(c, str) and c.strip()]
    result = {
        'status': REVIEW_STATUS,
        'checked_scope': CHECKED_SCOPE,
        'not_checked': list(NOT_CHECKED),
        'unreadable': len(texts) == 0,
        'ip_clause_count': 0,
        'flags': [],
    }
    if result['unreadable']:
        return result

    for clause_id, text in enumerate(texts):
        if not IP_TOPIC.search(text):
            continue
        result['ip_clause_count'] += 1

        issues = []
        for issue in ISSUE_LIBRARY:
            if not issue['pattern'].search(text):
                continue
            if issue.get('requires_no_payment_condition') and _payment_condition_present(text):
                continue
            issues.append({
                'id': issue['id'],
                'reason': issue['reason'],
                'paste_sentence': issue['paste_sentence'],
                'lawyer_when': issue['lawyer_when'],
            })

        if issues:
            result['flags'].append({
                'clause_id': clause_id,
                'clause_text': text,
                'issues': issues,
            })

    return result
