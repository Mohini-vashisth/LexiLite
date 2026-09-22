#!/usr/bin/env python3
"""
Performance benchmark for LexiLite REST API
Tests concurrent request handling and latency
"""

import os
import requests
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

API_URL = "http://localhost:8000/api/documents/analyze/"

# LEX-6: /analyze now requires authentication. Every user gets a token
# auto-created (see analyzer/signals.py) — grab yours from the Django
# shell: `from rest_framework.authtoken.models import Token;
# Token.objects.get(user__username='<you>').key`, then either export
# LEXILITE_API_TOKEN or pass --token.
API_TOKEN = os.getenv("LEXILITE_API_TOKEN")

# Sample legal text for testing
SAMPLE_TEXT = """
1. LIMITATION OF LIABILITY: In no event shall either party be liable for any indirect,
special, incidental, or consequential damages arising out of or in connection with this Agreement.

2. CONFIDENTIALITY: Each party agrees to maintain strict confidentiality regarding all
proprietary and sensitive information disclosed by the other party.

3. INDEMNIFICATION: Each party shall indemnify and hold harmless the other party from any
third-party claims arising from its breach of this Agreement.

4. TERMINATION: This Agreement may be terminated by either party with 30 days' written notice
to the other party. Upon termination, all obligations shall cease except those that survive termination.

5. ARBITRATION: Any disputes arising from this Agreement shall be resolved through binding arbitration
rather than litigation in courts of law.

6. FORCE MAJEURE: Neither party shall be liable for failure to perform due to circumstances beyond
its reasonable control, including natural disasters, wars, and government actions.

7. ENTIRE AGREEMENT: This Agreement, along with all exhibits and schedules, constitutes the entire
agreement between the parties and supersedes all prior negotiations and understandings.

8. AMENDMENTS: No amendment or modification of this Agreement shall be valid unless made in writing
and signed by authorized representatives of both parties.

9. SEVERABILITY: If any provision of this Agreement is found to be invalid or unenforceable, the
remaining provisions shall continue in full force and effect.

10. GOVERNING LAW: This Agreement shall be governed by and construed in accordance with the laws of
the jurisdiction specified herein, without regard to its conflict of law principles.
""" * 3  # Triple to get more clauses


def benchmark_single_request():
    """Test a single request"""
    try:
        headers = {"Authorization": f"Token {API_TOKEN}"} if API_TOKEN else {}
        start = time.time()
        response = requests.post(
            API_URL,
            json={"text": SAMPLE_TEXT, "filename": "test.pdf"},
            headers=headers,
            timeout=30
        )
        elapsed_ms = (time.time() - start) * 1000

        if response.status_code == 201:
            data = response.json()
            return {
                'status': 'success',
                'latency_ms': elapsed_ms,
                'api_time_ms': data.get('api_response_time_ms', 0),
                'risky_count': data.get('risky_count', 0),
            }
        else:
            return {
                'status': 'error',
                'latency_ms': elapsed_ms,
                'error': response.text,
            }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def run_benchmark(num_concurrent=10, num_requests=100):
    """Run concurrent benchmark"""
    if not API_TOKEN:
        print("❌ LEXILITE_API_TOKEN is not set — every request would fail with 403 "
              "since /analyze now requires authentication (LEX-6).")
        print("   Get your token: python manage.py shell -c \"from rest_framework.authtoken.models "
              "import Token; print(Token.objects.get(user__username='<you>').key)\"")
        print("   Then: export LEXILITE_API_TOKEN=<that key>")
        sys.exit(1)

    print(f"🧪 LexiLite REST API Benchmark")
    print(f"================================")
    print(f"📊 Configuration:")
    print(f"   Concurrent requests: {num_concurrent}")
    print(f"   Total requests: {num_requests}")
    print(f"   API URL: {API_URL}\n")

    results = []
    failed = 0

    print("⏳ Running benchmark (this may take a minute)...\n")

    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [
            executor.submit(benchmark_single_request)
            for _ in range(num_requests)
        ]

        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                results.append(result)

                if result['status'] == 'success':
                    print(f"✓ Request {i:3d}: {result['latency_ms']:6.1f}ms | API: {result['api_time_ms']:6.1f}ms | Risky: {result['risky_count']}")
                else:
                    print(f"✗ Request {i:3d}: ERROR - {result.get('error', 'Unknown')}")
                    failed += 1
            except Exception as e:
                print(f"✗ Request {i:3d}: FAILED - {e}")
                failed += 1

    # Analyze results
    successful = [r for r in results if r['status'] == 'success']
    latencies = [r['latency_ms'] for r in successful]
    api_times = [r['api_time_ms'] for r in successful]

    print(f"\n{'='*60}")
    print(f"📈 RESULTS")
    print(f"{'='*60}")
    print(f"Total Requests:        {num_requests}")
    print(f"Successful:            {len(successful)}")
    print(f"Failed:                {failed}")
    print(f"Success Rate:          {len(successful)/num_requests*100:.1f}%\n")

    if latencies:
        print(f"Total Latency (network + API):")
        print(f"  Min:                 {min(latencies):.1f}ms")
        print(f"  Max:                 {max(latencies):.1f}ms")
        print(f"  Mean:                {statistics.mean(latencies):.1f}ms")
        print(f"  Median (p50):        {statistics.median(latencies):.1f}ms")
        print(f"  P95:                 {sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")
        print(f"  P99:                 {sorted(latencies)[int(len(latencies)*0.99)]:.1f}ms")

        print(f"\nAPI Processing Time Only:")
        print(f"  Min:                 {min(api_times):.1f}ms")
        print(f"  Max:                 {max(api_times):.1f}ms")
        print(f"  Mean:                {statistics.mean(api_times):.1f}ms")
        print(f"  Median (p50):        {statistics.median(api_times):.1f}ms")
        print(f"  P95:                 {sorted(api_times)[int(len(api_times)*0.95)]:.1f}ms")
        print(f"  P99:                 {sorted(api_times)[int(len(api_times)*0.99)]:.1f}ms")

        print(f"\n{'='*60}")
        if statistics.mean(api_times) < 200:
            print(f"✅ PASS: Average API latency {statistics.mean(api_times):.1f}ms < 200ms")
        else:
            print(f"⚠️  WARNING: Average API latency {statistics.mean(api_times):.1f}ms >= 200ms")

        print(f"✅ Throughput: {len(successful) / (max(latencies)/1000):.1f} requests/sec")
        print(f"{'='*60}")
    else:
        print("❌ No successful requests")
        sys.exit(1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark LexiLite REST API')
    parser.add_argument('--concurrent', type=int, default=10, help='Concurrent requests')
    parser.add_argument('--total', type=int, default=100, help='Total requests')

    args = parser.parse_args()

    try:
        run_benchmark(num_concurrent=args.concurrent, num_requests=args.total)
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API at " + API_URL)
        print("   Make sure the server is running: python manage.py runserver")
        sys.exit(1)
