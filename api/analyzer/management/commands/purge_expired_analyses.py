"""
Deletes analyses older than the retention period (LEX-9).

Stored analyses contain the full text of confidential contracts, so they
are not kept forever. Run this on a schedule (daily is plenty):

    python manage.py purge_expired_analyses            # uses RETENTION_DAYS
    python manage.py purge_expired_analyses --days 7
    python manage.py purge_expired_analyses --dry-run  # only report
"""

from datetime import timedelta

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from analyzer.models import AnalysisResult


class Command(BaseCommand):
    help = "Delete analyses older than RETENTION_DAYS (0 = retention disabled)."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=None,
                            help="Override settings.RETENTION_DAYS for this run.")
        parser.add_argument("--dry-run", action="store_true",
                            help="Report how many rows would be deleted, delete nothing.")

    def handle(self, *args, **options):
        days = settings.RETENTION_DAYS if options["days"] is None else options["days"]
        if days <= 0:
            self.stdout.write("Retention disabled (days <= 0); nothing deleted.")
            return

        cutoff = timezone.now() - timedelta(days=days)
        expired = AnalysisResult.objects.filter(created_at__lt=cutoff)
        count = expired.count()

        if options["dry_run"]:
            self.stdout.write(f"[dry run] {count} analyses older than {days} days would be deleted.")
            return

        expired.delete()
        self.stdout.write(f"Deleted {count} analyses older than {days} days.")
