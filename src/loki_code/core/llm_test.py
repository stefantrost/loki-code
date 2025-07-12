"""
Ollama connection testing and validation for Loki Code.

This module provides comprehensive testing capabilities for Ollama connections,
model availability, and basic communication functionality.
"""

import json
import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

from ..utils.logging import get_logger, log_performance
from ..config.models import LokiCodeConfig


class TestResult(Enum):
    """Test result status enumeration."""
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class TestCase:
    """Individual test case result."""
    name: str
    status: TestResult
    message: str
    details: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None


@dataclass
class LLMTestReport:
    """Complete LLM test report."""
    overall_status: TestResult
    total_tests: int
    passed: int
    failed: int
    warnings: int
    skipped: int
    test_cases: List[TestCase]
    total_duration_ms: float
    recommendations: List[str]


class OllamaConnectionTester:
    """
    Comprehensive Ollama connection and functionality tester.
    
    This class provides methods to test various aspects of Ollama connectivity:
    - Service health checks
    - Model availability validation
    - Basic communication testing
    - Performance benchmarking
    """
    
    def __init__(self, config: LokiCodeConfig):
        """Initialize the Ollama tester with configuration.
        
        Args:
            config: LokiCodeConfig instance containing LLM settings
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.base_url = config.llm.base_url.rstrip('/')
        self.model = config.llm.model
        self.timeout = config.llm.timeout
        
        # Session for connection reuse
        self.session = requests.Session()
        self.session.timeout = self.timeout
        
        # Test results
        self.test_cases: List[TestCase] = []
        
    def run_comprehensive_test(self, verbose: bool = False) -> LLMTestReport:
        """Run a comprehensive test suite for Ollama connectivity.
        
        Args:
            verbose: Enable verbose logging and detailed output
            
        Returns:
            LLMTestReport with complete test results
        """
        start_time = time.perf_counter()
        self.test_cases = []
        
        self.logger.info("🔍 Starting comprehensive Ollama connection test...")
        
        if verbose:
            self.logger.info(f"Testing Ollama at: {self.base_url}")
            self.logger.info(f"Target model: {self.model}")
            self.logger.info(f"Timeout: {self.timeout}s")
        
        # Run test cases in order
        self._test_service_health(verbose)
        self._test_api_version(verbose)
        self._test_model_list(verbose)
        self._test_model_availability(verbose)
        self._test_basic_generation(verbose)
        
        # Calculate results
        total_duration = (time.perf_counter() - start_time) * 1000
        
        passed = sum(1 for tc in self.test_cases if tc.status == TestResult.SUCCESS)
        failed = sum(1 for tc in self.test_cases if tc.status == TestResult.FAILURE)
        warnings = sum(1 for tc in self.test_cases if tc.status == TestResult.WARNING)
        skipped = sum(1 for tc in self.test_cases if tc.status == TestResult.SKIPPED)
        
        # Determine overall status
        if failed > 0:
            overall_status = TestResult.FAILURE
        elif warnings > 0:
            overall_status = TestResult.WARNING
        else:
            overall_status = TestResult.SUCCESS
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        report = LLMTestReport(
            overall_status=overall_status,
            total_tests=len(self.test_cases),
            passed=passed,
            failed=failed,
            warnings=warnings,
            skipped=skipped,
            test_cases=self.test_cases,
            total_duration_ms=total_duration,
            recommendations=recommendations
        )
        
        self.logger.info(f"✅ Test suite completed in {total_duration:.1f}ms")
        return report
    
    def _test_service_health(self, verbose: bool = False) -> None:
        """Test if Ollama service is running and responding."""
        test_name = "Service Health Check"
        start_time = time.perf_counter()
        
        try:
            if verbose:
                self.logger.debug(f"Testing connection to {self.base_url}")
            
            with log_performance("Ollama health check", level=logging.DEBUG if verbose else logging.ERROR):
                response = self.session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            
            duration = (time.perf_counter() - start_time) * 1000
            
            if response.status_code == 200:
                self.test_cases.append(TestCase(
                    name=test_name,
                    status=TestResult.SUCCESS,
                    message="Ollama service is running and responsive",
                    details={"status_code": response.status_code, "response_time_ms": duration},
                    duration_ms=duration
                ))
                if verbose:
                    self.logger.debug(f"Service health check passed ({duration:.1f}ms)")
            else:
                self.test_cases.append(TestCase(
                    name=test_name,
                    status=TestResult.FAILURE,
                    message=f"Ollama service returned status {response.status_code}",
                    details={"status_code": response.status_code, "response": response.text[:200]},
                    duration_ms=duration
                ))
                
        except ConnectionError as e:
            duration = (time.perf_counter() - start_time) * 1000
            self.test_cases.append(TestCase(
                name=test_name,
                status=TestResult.FAILURE,
                message="Cannot connect to Ollama service (Connection refused)",
                details={"error": str(e), "base_url": self.base_url},
                duration_ms=duration
            ))
            self.logger.error(f"Connection failed: {e}")
            
        except Timeout as e:
            duration = (time.perf_counter() - start_time) * 1000
            self.test_cases.append(TestCase(
                name=test_name,
                status=TestResult.FAILURE,
                message=f"Connection timeout after {self.timeout}s",
                details={"error": str(e), "timeout": self.timeout},
                duration_ms=duration
            ))
            
        except RequestException as e:
            duration = (time.perf_counter() - start_time) * 1000
            self.test_cases.append(TestCase(
                name=test_name,
                status=TestResult.FAILURE,
                message=f"Request failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=duration
            ))
    
    def _test_api_version(self, verbose: bool = False) -> None:
        """Test Ollama API version information."""
        test_name = "API Version Check"
        start_time = time.perf_counter()
        
        # Skip if service health failed
        if any(tc.name == "Service Health Check" and tc.status == TestResult.FAILURE 
               for tc in self.test_cases):
            self.test_cases.append(TestCase(
                name=test_name,
                status=TestResult.SKIPPED,
                message="Skipped due to service health check failure"
            ))
            return
        
        try:
            response = self.session.get(f"{self.base_url}/api/version", timeout=self.timeout)
            duration = (time.perf_counter() - start_time) * 1000
            
            if response.status_code == 200:
                try:
                    version_info = response.json()
                    version = version_info.get('version', 'unknown')
                    self.test_cases.append(TestCase(
                        name=test_name,
                        status=TestResult.SUCCESS,
                        message=f"Ollama version: {version}",
                        details=version_info,
                        duration_ms=duration
                    ))
                    if verbose:
                        self.logger.debug(f"API version check passed: {version}")
                except json.JSONDecodeError:
                    self.test_cases.append(TestCase(
                        name=test_name,
                        status=TestResult.WARNING,
                        message="Version endpoint returned invalid JSON",
                        details={"response": response.text[:200]},
                        duration_ms=duration
                    ))
            else:
                self.test_cases.append(TestCase(
                    name=test_name,
                    status=TestResult.WARNING,
                    message=f"Version endpoint returned status {response.status_code}",
                    details={"status_code": response.status_code},
                    duration_ms=duration
                ))
                
        except Exception as e:
            duration = (time.perf_counter() - start_time) * 1000
            self.test_cases.append(TestCase(
                name=test_name,
                status=TestResult.WARNING,
                message=f"Version check failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=duration
            ))
    
    def _test_model_list(self, verbose: bool = False) -> None:
        """Test ability to list available models."""
        test_name = "Model List Retrieval"
        start_time = time.perf_counter()
        
        # Skip if service health failed
        if any(tc.name == "Service Health Check" and tc.status == TestResult.FAILURE 
               for tc in self.test_cases):
            self.test_cases.append(TestCase(
                name=test_name,
                status=TestResult.SKIPPED,
                message="Skipped due to service health check failure"
            ))
            return
        
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            duration = (time.perf_counter() - start_time) * 1000
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    models = data.get('models', [])
                    model_names = [model.get('name', 'unknown') for model in models]
                    
                    self.test_cases.append(TestCase(
                        name=test_name,
                        status=TestResult.SUCCESS,
                        message=f"Retrieved {len(models)} available models",
                        details={"model_count": len(models), "models": model_names[:10]},  # Limit for readability
                        duration_ms=duration
                    ))
                    
                    if verbose:
                        self.logger.debug(f"Available models: {', '.join(model_names[:5])}")
                        if len(model_names) > 5:
                            self.logger.debug(f"... and {len(model_names) - 5} more")
                            
                except json.JSONDecodeError as e:
                    self.test_cases.append(TestCase(
                        name=test_name,
                        status=TestResult.FAILURE,
                        message="Model list returned invalid JSON",
                        details={"error": str(e), "response": response.text[:200]},
                        duration_ms=duration
                    ))
            else:
                self.test_cases.append(TestCase(
                    name=test_name,
                    status=TestResult.FAILURE,
                    message=f"Model list request failed with status {response.status_code}",
                    details={"status_code": response.status_code},
                    duration_ms=duration
                ))
                
        except Exception as e:
            duration = (time.perf_counter() - start_time) * 1000
            self.test_cases.append(TestCase(
                name=test_name,
                status=TestResult.FAILURE,
                message=f"Failed to retrieve model list: {str(e)}",
                details={"error": str(e)},
                duration_ms=duration
            ))
    
    def _test_model_availability(self, verbose: bool = False) -> None:
        """Test if the configured model is available."""
        test_name = f"Model '{self.model}' Availability"
        start_time = time.perf_counter()
        
        # Skip if model list failed
        model_list_test = next((tc for tc in self.test_cases if tc.name == "Model List Retrieval"), None)
        if model_list_test and model_list_test.status == TestResult.FAILURE:
            self.test_cases.append(TestCase(
                name=test_name,
                status=TestResult.SKIPPED,
                message="Skipped due to model list retrieval failure"
            ))
            return
        
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            duration = (time.perf_counter() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                if self.model in model_names:
                    # Get model details
                    model_info = next((m for m in models if m.get('name') == self.model), {})
                    self.test_cases.append(TestCase(
                        name=test_name,
                        status=TestResult.SUCCESS,
                        message=f"Model '{self.model}' is available",
                        details={"model_info": model_info},
                        duration_ms=duration
                    ))
                    if verbose:
                        self.logger.debug(f"Target model '{self.model}' found")
                else:
                    # Check for partial matches (e.g., 'codellama:7b' vs 'codellama:7b-instruct')
                    partial_matches = [name for name in model_names if self.model.split(':')[0] in name]
                    
                    if partial_matches:
                        self.test_cases.append(TestCase(
                            name=test_name,
                            status=TestResult.WARNING,
                            message=f"Model '{self.model}' not found, but similar models available",
                            details={"requested_model": self.model, "similar_models": partial_matches},
                            duration_ms=duration
                        ))
                    else:
                        self.test_cases.append(TestCase(
                            name=test_name,
                            status=TestResult.FAILURE,
                            message=f"Model '{self.model}' is not available",
                            details={"requested_model": self.model, "available_models": model_names[:10]},
                            duration_ms=duration
                        ))
            else:
                duration = (time.perf_counter() - start_time) * 1000
                self.test_cases.append(TestCase(
                    name=test_name,
                    status=TestResult.FAILURE,
                    message=f"Failed to check model availability (status {response.status_code})",
                    details={"status_code": response.status_code},
                    duration_ms=duration
                ))
                
        except Exception as e:
            duration = (time.perf_counter() - start_time) * 1000
            self.test_cases.append(TestCase(
                name=test_name,
                status=TestResult.FAILURE,
                message=f"Model availability check failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=duration
            ))
    
    def _test_basic_generation(self, verbose: bool = False) -> None:
        """Test basic text generation with the configured model."""
        test_name = "Basic Text Generation"
        start_time = time.perf_counter()
        
        # Skip if model is not available
        model_test = next((tc for tc in self.test_cases if "Availability" in tc.name), None)
        if model_test and model_test.status == TestResult.FAILURE:
            self.test_cases.append(TestCase(
                name=test_name,
                status=TestResult.SKIPPED,
                message="Skipped due to model availability failure"
            ))
            return
        
        # Simple test prompt
        test_prompt = "Hello! Please respond with 'Connection test successful' to confirm you are working."
        
        try:
            payload = {
                "model": self.model,
                "prompt": test_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "max_tokens": 50,
                    "num_predict": 50
                }
            }
            
            if verbose:
                self.logger.debug(f"Sending test prompt to model '{self.model}'")
            
            with log_performance("Basic generation test", level=logging.DEBUG if verbose else logging.ERROR):
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=max(self.timeout, 30)  # Allow extra time for generation
                )
            
            duration = (time.perf_counter() - start_time) * 1000
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    generated_text = data.get('response', '').strip()
                    
                    if generated_text:
                        self.test_cases.append(TestCase(
                            name=test_name,
                            status=TestResult.SUCCESS,
                            message="Model generated response successfully",
                            details={
                                "prompt": test_prompt,
                                "response": generated_text[:100],  # Limit for readability
                                "response_length": len(generated_text),
                                "total_duration": data.get('total_duration'),
                                "load_duration": data.get('load_duration'),
                                "prompt_eval_duration": data.get('prompt_eval_duration'),
                                "eval_duration": data.get('eval_duration'),
                            },
                            duration_ms=duration
                        ))
                        if verbose:
                            self.logger.debug(f"Generated response: {generated_text[:50]}...")
                    else:
                        self.test_cases.append(TestCase(
                            name=test_name,
                            status=TestResult.WARNING,
                            message="Model responded but generated no text",
                            details={"response_data": data},
                            duration_ms=duration
                        ))
                        
                except json.JSONDecodeError as e:
                    self.test_cases.append(TestCase(
                        name=test_name,
                        status=TestResult.FAILURE,
                        message="Generation response contained invalid JSON",
                        details={"error": str(e), "response": response.text[:200]},
                        duration_ms=duration
                    ))
            else:
                self.test_cases.append(TestCase(
                    name=test_name,
                    status=TestResult.FAILURE,
                    message=f"Generation request failed with status {response.status_code}",
                    details={"status_code": response.status_code, "response": response.text[:200]},
                    duration_ms=duration
                ))
                
        except Timeout as e:
            duration = (time.perf_counter() - start_time) * 1000
            self.test_cases.append(TestCase(
                name=test_name,
                status=TestResult.FAILURE,
                message=f"Generation request timed out after {self.timeout}s",
                details={"error": str(e), "timeout": self.timeout},
                duration_ms=duration
            ))
            
        except Exception as e:
            duration = (time.perf_counter() - start_time) * 1000
            self.test_cases.append(TestCase(
                name=test_name,
                status=TestResult.FAILURE,
                message=f"Generation test failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=duration
            ))
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on test results."""
        recommendations = []
        
        # Check for service connection issues
        health_test = next((tc for tc in self.test_cases if tc.name == "Service Health Check"), None)
        if health_test and health_test.status == TestResult.FAILURE:
            if "Connection refused" in health_test.message:
                recommendations.append(
                    f"Ollama service is not running. Start it with: 'ollama serve' or check if it's running on {self.base_url}"
                )
            elif "timeout" in health_test.message.lower():
                recommendations.append(
                    f"Connection timeout - check if Ollama is running and accessible at {self.base_url}"
                )
        
        # Check for model availability issues
        model_test = next((tc for tc in self.test_cases if "Availability" in tc.name), None)
        if model_test and model_test.status == TestResult.FAILURE:
            recommendations.append(
                f"Model '{self.model}' is not available. Download it with: 'ollama pull {self.model}'"
            )
        elif model_test and model_test.status == TestResult.WARNING:
            if model_test.details and "similar_models" in model_test.details:
                similar = model_test.details["similar_models"]
                recommendations.append(
                    f"Consider using one of these available models: {', '.join(similar[:3])}"
                )
        
        # Check for generation issues
        gen_test = next((tc for tc in self.test_cases if tc.name == "Basic Text Generation"), None)
        if gen_test and gen_test.status == TestResult.FAILURE:
            if "timeout" in gen_test.message.lower():
                recommendations.append(
                    "Text generation timed out. Consider increasing the timeout in configuration or using a smaller model"
                )
        
        # General recommendations
        failed_count = sum(1 for tc in self.test_cases if tc.status == TestResult.FAILURE)
        if failed_count == 0:
            recommendations.append("✅ All tests passed! Your Ollama setup is working correctly.")
        elif failed_count > 2:
            recommendations.append(
                "Multiple tests failed. Check Ollama installation and ensure the service is running properly"
            )
        
        return recommendations
    
    def __del__(self):
        """Clean up the requests session."""
        if hasattr(self, 'session'):
            self.session.close()


def format_test_report(report: LLMTestReport, verbose: bool = False) -> str:
    """Format a test report for console output.
    
    Args:
        report: LLMTestReport to format
        verbose: Include detailed information
        
    Returns:
        Formatted string representation of the report
    """
    lines = []
    
    # Header
    if report.overall_status == TestResult.SUCCESS:
        lines.append("🎉 LLM Connection Test: SUCCESS")
    elif report.overall_status == TestResult.WARNING:
        lines.append("⚠️  LLM Connection Test: WARNING")
    else:
        lines.append("❌ LLM Connection Test: FAILED")
    
    # Summary
    lines.append(f"📊 Results: {report.passed}✅ {report.failed}❌ {report.warnings}⚠️  {report.skipped}⏭️")
    lines.append(f"⏱️  Total time: {report.total_duration_ms:.1f}ms")
    lines.append("")
    
    # Individual test results
    for test_case in report.test_cases:
        if test_case.status == TestResult.SUCCESS:
            icon = "✅"
        elif test_case.status == TestResult.WARNING:
            icon = "⚠️ "
        elif test_case.status == TestResult.SKIPPED:
            icon = "⏭️ "
        else:
            icon = "❌"
        
        duration_str = f" ({test_case.duration_ms:.1f}ms)" if test_case.duration_ms else ""
        lines.append(f"{icon} {test_case.name}: {test_case.message}{duration_str}")
        
        # Add details in verbose mode
        if verbose and test_case.details:
            for key, value in test_case.details.items():
                if key in ['error', 'models', 'similar_models', 'available_models']:
                    lines.append(f"   {key}: {value}")
    
    # Recommendations
    if report.recommendations:
        lines.append("")
        lines.append("💡 Recommendations:")
        for rec in report.recommendations:
            lines.append(f"   • {rec}")
    
    return "\n".join(lines)


def test_ollama_connection(config: LokiCodeConfig, verbose: bool = False) -> LLMTestReport:
    """
    Test Ollama connection with the given configuration.
    
    Args:
        config: LokiCodeConfig instance
        verbose: Enable verbose logging and output
        
    Returns:
        LLMTestReport with test results
    """
    tester = OllamaConnectionTester(config)
    return tester.run_comprehensive_test(verbose=verbose)