#!/usr/bin/env python3
"""
Test FastAPI Prometheus metrics integration
Validates comprehensive instrumentation from your production API
"""
import asyncio
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()


class FastAPIMetricsTester:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.metrics_url = f"{api_base_url}/metrics"
        self.expected_metrics = [
            "http_requests_total",  # Updated: was fastapi_requests_total
            "http_request_duration_seconds",  # Updated: was fastapi_request_duration_seconds
            "http_request_size_bytes",  # New: request size tracking
            "python_info",
            "process_resident_memory_bytes",
            "process_virtual_memory_bytes",
            "process_cpu_seconds_total",
            "up"
        ]

    async def test_api_health(self) -> bool:
        """Test basic API health"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_base_url}/health")
                if response.status_code == 200:
                    console.print("✅ API health check passed", style="green")
                    return True
                else:
                    console.print(f"❌ API health failed: {response.status_code}", style="red")
                    return False
        except Exception as e:
            console.print(f"❌ API connection failed: {e}", style="red")
            return False

    async def fetch_metrics(self) -> Optional[str]:
        """Fetch Prometheus metrics from /metrics endpoint"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.metrics_url, timeout=10.0)
                if response.status_code == 200:
                    console.print("✅ Metrics endpoint accessible", style="green")
                    return response.text
                else:
                    console.print(f"❌ Metrics endpoint failed: {response.status_code}", style="red")
                    return None
        except Exception as e:
            console.print(f"❌ Metrics fetch failed: {e}", style="red")
            return None

    def parse_metrics(self, metrics_text: str) -> Dict[str, List[str]]:
        """Parse Prometheus metrics text"""
        metric_families = {}

        for line in metrics_text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Extract metric name
            match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)', line)
            if match:
                metric_name = match.group(1)
                if metric_name not in metric_families:
                    metric_families[metric_name] = []
                metric_families[metric_name].append(line)

        return metric_families

    def validate_fastapi_instrumentation(self, metrics: Dict[str, List[str]]) -> Dict[str, bool]:
        """Validate FastAPI Prometheus instrumentation"""
        validation_results = {}

        for expected_metric in self.expected_metrics:
            found = any(expected_metric in metric_name for metric_name in metrics.keys())
            validation_results[expected_metric] = found

            if found:
                console.print(f"✅ Found: {expected_metric}", style="green")
            else:
                console.print(f"❌ Missing: {expected_metric}", style="red")

        return validation_results

    def analyze_fastapi_specific_metrics(self, metrics: Dict[str, List[str]]):
        """Analyze FastAPI-specific metrics in detail"""
        console.print("\n🔍 FastAPI Metrics Analysis (Updated Names):", style="bold blue")

        # Request metrics (updated naming)
        if 'http_requests_total' in metrics:
            requests_total = metrics['http_requests_total']
            console.print(f"📊 HTTP Request Total Metrics: {len(requests_total)} series", style="cyan")

            # Show sample requests
            for i, request_metric in enumerate(requests_total[:3]):
                console.print(f"   [{i + 1}] {request_metric}", style="yellow")

        # Duration metrics (histograms)
        duration_metrics = [k for k in metrics.keys() if 'duration_seconds' in k and 'http' in k]
        if duration_metrics:
            console.print(f"⏱️ Duration Metrics: {len(duration_metrics)} types", style="cyan")
            for metric in duration_metrics[:5]:
                console.print(f"   • {metric}", style="yellow")

        # Size metrics
        if 'http_request_size_bytes' in metrics:
            size_metrics = metrics['http_request_size_bytes']
            console.print(f"📏 Request Size Metrics: {len(size_metrics)} series", style="cyan")
            for metric in size_metrics[:2]:
                console.print(f"   {metric}", style="yellow")

    async def generate_test_traffic(self):
        """Generate test traffic to create metrics"""
        console.print("\n🚗 Generating test traffic...", style="blue")

        endpoints = [
            "/",
            "/health",
            "/chat",  # This might fail, but will create metrics
            "/docs",
            "/metrics"
        ]

        async with httpx.AsyncClient() as client:
            for endpoint in endpoints:
                try:
                    url = f"{self.api_base_url}{endpoint}"
                    if endpoint == "/chat":
                        # POST request for chat
                        response = await client.post(url, json={"query": "test"}, timeout=5.0)
                    else:
                        # GET request for others
                        response = await client.get(url, timeout=5.0)

                    console.print(f"   {endpoint}: {response.status_code}", style="cyan")

                except Exception as e:
                    console.print(f"   {endpoint}: ERROR - {e}", style="red")

                # Small delay between requests
                await asyncio.sleep(0.5)

    def display_metrics_summary(self, metrics: Dict[str, List[str]], validation: Dict[str, bool]):
        """Display comprehensive metrics summary"""
        table = Table(title="FastAPI Prometheus Metrics Validation")
        table.add_column("Metric Type", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Count", style="yellow")
        table.add_column("Sample", style="green")

        for metric_name, is_valid in validation.items():
            status = "✅ Found" if is_valid else "❌ Missing"
            count = len([k for k in metrics.keys() if metric_name in k])

            # Get a sample metric line
            sample = "N/A"
            if is_valid and count > 0:
                matching_metrics = [k for k in metrics.keys() if metric_name in k]
                if matching_metrics and metrics[matching_metrics[0]]:
                    sample = metrics[matching_metrics[0]][0][:50] + "..."

            table.add_row(metric_name, status, str(count), sample)

        console.print(table)

    def check_instrumentation_quality(self, metrics: Dict[str, List[str]]) -> Dict[str, str]:
        """Check quality of FastAPI instrumentation"""
        quality_checks = {}

        # Check for request labels
        http_requests = [k for k in metrics.keys() if 'http_requests_total' in k]
        if http_requests:
            sample_line = metrics[http_requests[0]][0] if metrics[http_requests[0]] else ""

            if 'method=' in sample_line:
                quality_checks['method_labels'] = "✅ HTTP methods tracked"
            else:
                quality_checks['method_labels'] = "❌ HTTP methods not tracked"

            if 'handler=' in sample_line or 'path=' in sample_line:
                quality_checks['endpoint_labels'] = "✅ Endpoints tracked"
            else:
                quality_checks['endpoint_labels'] = "❌ Endpoints not tracked"

            if 'status=' in sample_line:
                quality_checks['status_labels'] = "✅ Status codes tracked"
            else:
                quality_checks['status_labels'] = "❌ Status codes not tracked"

        # Check for histogram buckets
        duration_buckets = [k for k in metrics.keys() if 'duration_seconds_bucket' in k]
        if duration_buckets:
            quality_checks['histogram_buckets'] = f"✅ {len(duration_buckets)} histogram buckets"
        else:
            quality_checks['histogram_buckets'] = "❌ No histogram buckets found"

        # Check for system metrics
        system_metrics = [k for k in metrics.keys() if any(x in k for x in ['process_', 'python_'])]
        if system_metrics:
            quality_checks['system_metrics'] = f"✅ {len(system_metrics)} system metrics"
        else:
            quality_checks['system_metrics'] = "❌ No system metrics found"

        return quality_checks


async def main():
    """Main test execution"""
    console.print("🎯 FastAPI Prometheus Metrics Comprehensive Test", style="bold blue")
    console.print("=" * 65)

    tester = FastAPIMetricsTester()

    # 1. API Health Check
    console.print("\n1️⃣ API Health Check")
    api_healthy = await tester.test_api_health()

    if not api_healthy:
        console.print("\n❌ API not available - cannot test metrics", style="red")
        console.print("💡 Start the API: docker compose -f docker-compose.dev.yml up api", style="yellow")
        return

    # 2. Generate test traffic
    await tester.generate_test_traffic()

    # 3. Fetch metrics
    console.print("\n2️⃣ Fetching Prometheus Metrics")
    metrics_text = await tester.fetch_metrics()

    if not metrics_text:
        console.print("❌ Could not fetch metrics", style="red")
        return

    console.print(f"📊 Retrieved {len(metrics_text.split())} lines of metrics", style="green")

    # 4. Parse metrics
    console.print("\n3️⃣ Parsing Metrics")
    parsed_metrics = tester.parse_metrics(metrics_text)
    console.print(f"🔍 Found {len(parsed_metrics)} metric families", style="cyan")

    # 5. Validate FastAPI instrumentation
    console.print("\n4️⃣ Validating FastAPI Instrumentation")
    validation_results = tester.validate_fastapi_instrumentation(parsed_metrics)

    valid_count = sum(validation_results.values())
    total_count = len(validation_results)
    console.print(f"\n📈 Validation Summary: {valid_count}/{total_count} expected metrics found",
                  style="green" if valid_count > total_count * 0.8 else "yellow")

    # 6. Analyze FastAPI-specific metrics
    tester.analyze_fastapi_specific_metrics(parsed_metrics)

    # 7. Check instrumentation quality
    console.print("\n5️⃣ Instrumentation Quality Check")
    quality_checks = tester.check_instrumentation_quality(parsed_metrics)

    for check_name, result in quality_checks.items():
        console.print(f"   {check_name}: {result}", style="cyan")

    # 8. Display comprehensive summary
    console.print("\n6️⃣ Comprehensive Metrics Summary")
    tester.display_metrics_summary(parsed_metrics, validation_results)

    # 9. Grafana dashboard readiness - FIXED VERSION
    console.print("\n7️⃣ Grafana Dashboard Readiness", style="bold blue")

    required_for_dashboard = [
        'http_requests_total',
        'http_request_duration_seconds_bucket',
        'process_resident_memory_bytes',
        'process_cpu_seconds_total'
    ]

    dashboard_ready = all(
        any(req in metric for metric in parsed_metrics.keys())
        for req in required_for_dashboard
    )

    if dashboard_ready:
        console.print("✅ All required metrics available for Grafana dashboard", style="green")
        console.print("🎉 Ready to import modern HTTP metrics dashboard", style="green")
    else:
        missing = [req for req in required_for_dashboard
                   if not any(req in metric for metric in parsed_metrics.keys())]
        console.print(f"❌ Missing metrics for dashboard: {missing}", style="red")

    # 10. Sample Prometheus queries
    console.print("\n8️⃣ Sample Prometheus Queries for Your Dashboard", style="bold blue")

    sample_queries = [
        ("Request Rate", "rate(http_requests_total[5m])"),
        ("95th Percentile Latency", "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"),
        ("Error Rate", "rate(http_requests_total{status!~\"2..\"}[5m])"),
        ("Memory Usage", "process_resident_memory_bytes"),
        ("CPU Usage", "rate(process_cpu_seconds_total[5m]) * 100"),
        ("Request Size", "rate(http_request_size_bytes_sum[5m])")
    ]

    query_table = Table(title="Prometheus Queries for Grafana")
    query_table.add_column("Metric", style="cyan")
    query_table.add_column("Prometheus Query", style="yellow")

    for metric_name, query in sample_queries:
        query_table.add_row(metric_name, query)

    console.print(query_table)

    console.print("\n✅ FastAPI Metrics Test Complete!", style="bold green")
    console.print("🎯 Next: Set up Prometheus + Grafana to visualize these metrics", style="blue")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n⏹️ Test interrupted", style="yellow")
    except Exception as e:
        console.print(f"\n💥 Test failed: {e}", style="red")
        import traceback

        traceback.print_exc()
