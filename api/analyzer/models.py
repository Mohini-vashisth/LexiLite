from django.conf import settings
from django.db import models
from django.utils import timezone

class AnalysisResult(models.Model):
    RISK_CHOICES = [
        ('safe', 'Safe'),
        ('risky', 'Risky'),
    ]

    # Every analysis belongs to whoever ran it (LEX-6) — required, not
    # nullable, since /analyze and /upload now require authentication, so
    # every new row always has a real requesting user.
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='analyses')
    document_id = models.CharField(max_length=255, unique=True)
    filename = models.CharField(max_length=255)
    clauses = models.JSONField()
    analysis = models.JSONField()
    processing_time_ms = models.FloatField()
    model_used = models.CharField(max_length=50, choices=[('sbert', 'Sentence-BERT'), ('roberta', 'RoBERTa')])
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['document_id']),
            models.Index(fields=['-created_at']),
            models.Index(fields=['owner', '-created_at']),
        ]

    def __str__(self):
        return f"{self.filename} - {self.document_id}"
