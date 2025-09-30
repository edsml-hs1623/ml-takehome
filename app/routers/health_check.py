# Health check router - run tests and return results
import subprocess
import json
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

class TestResult(BaseModel):
    test_name: str
    status: str  # "PASSED", "FAILED", "ERROR"
    duration: Optional[float] = None
    error_message: Optional[str] = None

class HealthCheckResponse(BaseModel):
    status: str  # "healthy", "unhealthy"
    total_tests: int
    passed_tests: int
    failed_tests: int
    execution_time: float
    test_results: List[TestResult]
    summary: str

def parse_pytest_output(output: str) -> tuple[List[TestResult], dict]:
    """Parse pytest output to extract test results"""
    test_results = []
    summary = {}
    
    # Parse individual test results - improved regex to handle progress percentages
    test_pattern = r'tests/([^:]+)::([^\s]+)\s+([A-Z]+)(?:\s+\[([^\]]+)\])?'
    for line in output.split('\n'):
        # Skip lines with progress percentages
        if '%' in line and '[' in line and ']' in line:
            continue
            
        match = re.search(test_pattern, line)
        if match:
            file_name, test_name, status, duration_str = match.groups()
            try:
                duration = float(duration_str) if duration_str else None
            except ValueError:
                duration = None
            
            test_results.append(TestResult(
                test_name=f"{file_name}::{test_name}",
                status=status,
                duration=duration
            ))
    
    # Parse summary - improved regex to handle various formats
    summary_patterns = [
        r'(\d+)\s+passed(?:,\s+(\d+)\s+failed)?(?:,\s+(\d+)\s+warnings)?\s+in\s+([\d.]+)s',
        r'(\d+)\s+passed(?:,\s+(\d+)\s+failed)?(?:,\s+(\d+)\s+warnings)?',
        r'(\d+)\s+passed'
    ]
    
    for pattern in summary_patterns:
        summary_match = re.search(pattern, output)
        if summary_match:
            passed = int(summary_match.group(1))
            failed = int(summary_match.group(2) or 0) if summary_match.groups()[1] else 0
            warnings = int(summary_match.group(3) or 0) if len(summary_match.groups()) > 2 and summary_match.groups()[2] else 0
            execution_time = float(summary_match.group(4)) if len(summary_match.groups()) > 3 and summary_match.groups()[3] else 0.0
            
            summary = {
                "total_tests": passed + failed,
                "passed_tests": passed,
                "failed_tests": failed,
                "warnings": warnings,
                "execution_time": execution_time
            }
            break
    
    return test_results, summary

@router.get("/", response_model=HealthCheckResponse)
async def health_check():
    """
    Run the full test suite and return health status
    """
    try:
        # Run pytest with deprecation warnings ignored
        cmd = [
            "python", "-m", "pytest", 
            "-W", "ignore::DeprecationWarning",
            "--tb=short",
            "-v"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        # Parse the output
        test_results, summary = parse_pytest_output(result.stdout)
        
        # Determine overall health status
        if result.returncode == 0 and summary.get("failed_tests", 0) == 0:
            status = "healthy"
        else:
            status = "unhealthy"
        
        # Create response
        response = HealthCheckResponse(
            status=status,
            total_tests=summary.get("total_tests", 0),
            passed_tests=summary.get("passed_tests", 0),
            failed_tests=summary.get("failed_tests", 0),
            execution_time=summary.get("execution_time", 0.0),
            test_results=test_results,
            summary=f"Tests: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)} passed, "
                   f"Execution time: {summary.get('execution_time', 0.0):.2f}s"
        )
        
        return response
        
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=408, 
            detail="Test execution timed out after 2 minutes"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to run health check: {str(e)}"
        )
