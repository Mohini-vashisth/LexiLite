"""
Auto-creates a DRF auth Token for every new User, so a fresh account is
immediately usable for API-key-style access (Authorization: Token <key>)
without a separate manual "generate my token" step.
"""

from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver
from rest_framework.authtoken.models import Token


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_auth_token(sender, instance, created, **kwargs):
    if created:
        Token.objects.create(user=instance)
