"""
Tests for the network-free parts of risk_pipeline.py: quote checking, form
validation and the playbook rules. Run from the repo root:

    python3 -m unittest discover -s train_model -p "test_risk_pipeline.py"
"""
import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rp", os.path.join(HERE, "risk_pipeline.py"))
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

IP, TFC = rp.IP, rp.TFC


def form(clause_type, **values):
    """A validated form with the given values (quotes are placeholders); every other field missing."""
    out = {name: {"value": None, "quote": ""} for name in rp.FORMS[clause_type]}
    for name, value in values.items():
        out[name] = {"value": value, "quote": f"<{name}>"}
    return out


class QuoteTests(unittest.TestCase):
    def test_quote_must_appear_in_the_clause(self):
        clause = "The Client may terminate this Agreement on thirty (30) days’ notice."
        self.assertTrue(rp.quote_in_clause("thirty (30) days' notice", clause))
        self.assertTrue(rp.quote_in_clause("the client MAY   terminate", clause))
        self.assertFalse(rp.quote_in_clause("sixty (60) days' notice", clause))
        self.assertFalse(rp.quote_in_clause("", clause))

    def test_invented_quote_drops_the_answer(self):
        clause = "The Client may terminate this Agreement at any time."
        raw = {"who_can_terminate": {"value": "client_only", "quote": "The Client may terminate"},
               "notice_required": {"value": "yes", "quote": "on 30 days notice"},
               "paid_for_work_done": {"value": "not_mentioned", "quote": ""}}
        validated, problems = rp.validate_form(TFC, raw, clause)
        self.assertEqual(validated["who_can_terminate"]["value"], "client_only")
        self.assertIsNone(validated["notice_required"]["value"])
        self.assertEqual(validated["paid_for_work_done"]["value"], "not_mentioned")
        self.assertIn(("notice_required", "quote not found in the clause"), problems)

    def test_answer_outside_the_list_is_dropped(self):
        validated, problems = rp.validate_form(IP, {"scope": {"value": "everything", "quote": "x"}}, "text")
        self.assertIsNone(validated["scope"]["value"])
        self.assertTrue(any("outside the allowed list" in p for _, p in problems))

    def test_notice_days_must_match_the_quote(self):
        clause = "Either party may terminate on two weeks' notice or thirty (30) days in writing."
        ok, _ = rp.validate_form(TFC, {"notice_days": {"value": 14, "quote": "two weeks' notice"}}, clause)
        self.assertEqual(ok["notice_days"]["value"], 14)
        ok, _ = rp.validate_form(TFC, {"notice_days": {"value": 30, "quote": "thirty (30) days"}}, clause)
        self.assertEqual(ok["notice_days"]["value"], 30)
        bad, problems = rp.validate_form(TFC, {"notice_days": {"value": 60, "quote": "thirty (30) days"}}, clause)
        self.assertIsNone(bad["notice_days"]["value"])
        self.assertTrue(problems)


class IpRuleTests(unittest.TestCase):
    def test_licence_only_is_low(self):
        self.assertEqual(rp.assess(IP, form(IP, transfers_ownership="no", moral_rights_waived="no"))["risk"], "low")

    def test_licence_only_with_moral_rights_unread_is_unsure(self):
        self.assertEqual(rp.assess(IP, form(IP, transfers_ownership="no"))["risk"], "unsure")

    def test_text_that_transfers_nothing_is_unsure(self):
        self.assertEqual(rp.assess(IP, form(IP, transfers_ownership="not_stated"))["risk"], "unsure")

    def test_project_deliverables_on_payment_is_medium(self):
        result = rp.assess(IP, form(IP, transfers_ownership="yes", scope="project_deliverables",
                                    transfer_timing="on_full_payment", pre_existing_materials="kept_by_freelancer"))
        self.assertEqual(result["risk"], "medium")
        self.assertEqual(result["playbook_position"], "acceptable")

    def test_blanket_scope_is_high_even_with_other_facts_missing(self):
        result = rp.assess(IP, form(IP, scope="all_work_during_engagement"))
        self.assertEqual(result["risk"], "high")
        self.assertEqual(result["reasons"][0]["quote"], "<scope>")

    def test_before_payment_and_moral_rights_are_high(self):
        result = rp.assess(IP, form(IP, transfers_ownership="yes", scope="project_deliverables",
                                    transfer_timing="before_payment", moral_rights_waived="yes"))
        self.assertEqual(result["risk"], "high")
        self.assertEqual({r["rule"] for r in result["reasons"]}, {"ip_before_payment", "ip_moral_rights_waiver"})

    def test_missing_facts_give_unsure(self):
        result = rp.assess(IP, form(IP))
        self.assertEqual(result["risk"], "unsure")
        self.assertIsNone(result["playbook_position"])
        result = rp.assess(IP, form(IP, transfers_ownership="yes", scope="project_deliverables"))
        self.assertEqual(result["risk"], "unsure")

    def test_every_result_carries_the_disclaimer(self):
        result = rp.assess(IP, form(IP, transfers_ownership="no", moral_rights_waived="no"))
        self.assertEqual(result["playbook_status"], "draft_not_lawyer_reviewed")
        self.assertIn("can be wrong", result["disclaimer"])


class TerminationRuleTests(unittest.TestCase):
    def test_mutual_with_notice_and_payment_is_low(self):
        facts = dict(who_can_terminate="both", notice_required="yes", notice_days=15, paid_for_work_done="yes",
                     paid_for_work_in_progress="yes", waives_other_claims="no", penalty_on_freelancer="no")
        self.assertEqual(rp.assess(TFC, form(TFC, **facts))["risk"], "low")

    def test_a_missing_risk_fact_never_gives_low(self):
        facts = dict(who_can_terminate="both", notice_required="yes", notice_days=15, paid_for_work_done="yes",
                     waives_other_claims="no")   # penalty_on_freelancer unread
        self.assertEqual(rp.assess(TFC, form(TFC, **facts))["risk"], "unsure")

    def test_client_only_with_notice_and_payment_is_medium(self):
        result = rp.assess(TFC, form(TFC, who_can_terminate="client_only", notice_required="yes", notice_days=30,
                                     paid_for_work_done="yes"))
        self.assertEqual(result["risk"], "medium")

    def test_no_notice_or_no_payment_is_high(self):
        self.assertEqual(rp.assess(TFC, form(TFC, notice_required="no"))["risk"], "high")
        self.assertEqual(rp.assess(TFC, form(TFC, paid_for_work_done="no"))["risk"], "high")

    def test_freelancer_only_is_low_unless_it_costs_them(self):
        self.assertEqual(rp.assess(TFC, form(TFC, who_can_terminate="freelancer_only",
                                             penalty_on_freelancer="not_mentioned"))["risk"], "low")
        self.assertEqual(rp.assess(TFC, form(TFC, who_can_terminate="freelancer_only"))["risk"], "unsure")
        result = rp.assess(TFC, form(TFC, who_can_terminate="freelancer_only", penalty_on_freelancer="yes"))
        self.assertEqual(result["risk"], "high")

    def test_payment_not_mentioned_is_medium_not_low(self):
        result = rp.assess(TFC, form(TFC, who_can_terminate="both", notice_required="yes", notice_days=90,
                                     paid_for_work_done="not_mentioned"))
        self.assertEqual(result["risk"], "medium")
        self.assertIn("tfc_payment_unstated", {r["rule"] for r in result["reasons"]})

    def test_missing_who_gives_unsure(self):
        result = rp.assess(TFC, form(TFC, notice_required="yes", notice_days=30, paid_for_work_done="yes"))
        self.assertEqual(result["risk"], "unsure")


if __name__ == "__main__":
    unittest.main()
