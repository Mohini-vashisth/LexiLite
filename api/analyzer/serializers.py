from rest_framework import serializers
from .models import AnalysisResult


class AnalysisResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisResult
        fields = [
            'id', 'document_id', 'filename', 'clauses', 'analysis',
            'processing_time_ms', 'model_used', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
