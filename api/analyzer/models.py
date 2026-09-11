from django.db import models
from django.utils import timezone

class AnalysisResult(models.Model):
    RISK_CHOICES = [
        ('safe', 'Safe'),
        ('risky', 'Risky'),
    ]

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
        ]

    def __str__(self):
        return f"{self.filename} - {self.document_id}"
