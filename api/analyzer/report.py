"""
The paid full report: plain-language explanations for the IP flags, written by
a self-hosted model.

Privacy: the model runs on our own server (LLM_ENDPOINT). Contract text is sent
only to that endpoint, never to a third-party AI service. If LLM_ENDPOINT is not
set, the report is not available at all. It never falls back to a third party.

Safety: the model only rewrites explanations for flags the reviewed library
already produced. It cannot add a flag. If the model is unreachable, returns
nothing, or returns something that is not text, that flag keeps the library's
reason and wording. Every explanation is labelled as pending lawyer review.
"""
import logging

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

EXPLANATION_STATUS = 'ai_explanation_pending_lawyer_review'
MAX_EXPLANATION_CHARS = 800
REQUEST_TIMEOUT_SECONDS = 60


class ReportUnavailable(Exception):
    """Raised when the paid report cannot be produced at all (no model configured)."""


def llm_configured():
    return bool(getattr(settings, 'LLM_ENDPOINT', ''))


def _ask_model(clause_text, issue):
    """Ask the self-hosted model to explain one flag in plain language.
    Returns the explanation text, or None if the model could not answer."""
    prompt = (
        "You explain contract risks to a freelancer in plain English. Do not add any risk "
        "that is not described below. Do not say the clause is safe. Keep it under 120 words.\n\n"
        f"Clause: {clause_text}\n"
        f"Risk already identified: {issue['reason']}\n"
        f"Suggested sentence to send the client: {issue['paste_sentence']}\n"
        "Explain what the clause means for the freelancer and why the suggested sentence helps."
    )
    try:
        response = requests.post(
            settings.LLM_ENDPOINT,
            json={'model': settings.LLM_MODEL, 'prompt': prompt, 'stream': False},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        text = response.json().get('response')
    except (requests.RequestException, ValueError):
        logger.exception("Report model call failed")
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    return text.strip()[:MAX_EXPLANATION_CHARS]


def build_full_report(review):
    """
    Turn an IP review (from ip_review.review_ip_clauses) into the paid report.

    Raises ReportUnavailable if no model endpoint is configured.
    Every flag gets an explanation: the model's if it answered, otherwise the
    library's reason, and the explanation_source field says which.
    """
    if not llm_configured():
        raise ReportUnavailable('The full report is not available on this server.')

    flags = []
    for flag in review['flags']:
        issues = []
        for issue in flag['issues']:
            explanation = _ask_model(flag['clause_text'], issue)
            issues.append({
                **issue,
                'explanation': explanation or issue['reason'],
                'explanation_source': 'model' if explanation else 'library',
            })
        flags.append({**flag, 'issues': issues})

    return {
        **review,
        'report_status': EXPLANATION_STATUS,
        'flags': flags,
    }
