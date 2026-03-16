"""
Knowledge Validator for Triton Knowledge Base

Provides validation functionality for knowledge entries.

Usage:
    python knowledge_validator.py --input <file> --type case|rule|document
    python knowledge_validator.py --input <json_string> --type case|rule
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class ValidationLevel(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    valid: bool
    level: ValidationLevel
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "level": self.level.value,
            "message": self.message,
            "field": self.field,
            "suggestion": self.suggestion,
        }


class KnowledgeValidator:
    CASE_TYPES = ["precision_issue", "optimization", "conversion", "compilation_error", "runtime_error"]
    RULE_TYPES = ["hardware_constraint", "performance_rule", "best_practice", "api_limitation"]
    SEVERITY_LEVELS = ["high", "medium", "low"]
    
    def __init__(self, knowledge_base_path: str = None):
        self.knowledge_base_path = Path(knowledge_base_path) if knowledge_base_path else None
        self.existing_ids: set = set()
        
        if self.knowledge_base_path:
            self._load_existing_ids()
    
    def _load_existing_ids(self):
        cases_dir = self.knowledge_base_path / "cases"
        if cases_dir.exists():
            for case_file in cases_dir.glob("**/*.json"):
                try:
                    with open(case_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "case_id" in data:
                        self.existing_ids.add(data["case_id"])
                except Exception:
                    pass
        
        rules_dir = self.knowledge_base_path / "rules"
        if rules_dir.exists():
            for rule_file in rules_dir.glob("*.json"):
                try:
                    with open(rule_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "rule_id" in data:
                        self.existing_ids.add(data["rule_id"])
                except Exception:
                    pass
    
    def validate(self, data: Dict[str, Any], knowledge_type: str) -> Dict[str, Any]:
        if knowledge_type == "case":
            results = self.validate_case(data)
        elif knowledge_type == "rule":
            results = self.validate_rule(data)
        elif knowledge_type == "document":
            results = self.validate_document(data.get("content", ""), data.get("name"))
        else:
            return {
                "valid": False,
                "errors": [{"message": f"Unknown knowledge type: {knowledge_type}"}]
            }
        
        return self.get_validation_summary(results)
    
    def validate_case(self, case_data: Dict[str, Any]) -> List[ValidationResult]:
        results = []
        
        required_fields = ["case_id", "case_type", "title", "problem", "solution"]
        for field in required_fields:
            if field not in case_data:
                results.append(ValidationResult(
                    valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Missing required field: {field}",
                    field=field,
                    suggestion=f"Add the '{field}' field",
                ))
        
        if "case_id" in case_data:
            id_result = self._validate_case_id(case_data["case_id"])
            results.extend(id_result)
        
        if "case_type" in case_data and case_data["case_type"] not in self.CASE_TYPES:
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.WARNING,
                message=f"Unknown case_type: {case_data['case_type']}",
                field="case_type",
                suggestion=f"Use one of: {', '.join(self.CASE_TYPES)}",
            ))
        
        if "problem" in case_data:
            results.extend(self._validate_problem(case_data["problem"]))
        
        if "solution" in case_data:
            results.extend(self._validate_solution(case_data["solution"]))
        
        if "metadata" in case_data:
            results.extend(self._validate_metadata(case_data["metadata"]))
        
        return results
    
    def _validate_case_id(self, case_id: str) -> List[ValidationResult]:
        results = []
        
        if not case_id:
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message="case_id cannot be empty",
                field="case_id",
            ))
            return results
        
        if not re.match(r'^[a-z][a-z0-9_]*$', case_id):
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message="case_id must start with lowercase letter and contain only lowercase letters, numbers, and underscores",
                field="case_id",
                suggestion="Use format: descriptive_name_YYYYMMDD",
            ))
        
        if len(case_id) > 100:
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.WARNING,
                message="case_id is too long (max 100 characters)",
                field="case_id",
            ))
        
        if case_id in self.existing_ids:
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.WARNING,
                message="case_id already exists",
                field="case_id",
                suggestion="Use a unique case_id",
            ))
        
        return results
    
    def _validate_problem(self, problem: Any) -> List[ValidationResult]:
        results = []
        
        if not isinstance(problem, dict):
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message="problem must be a dictionary",
                field="problem",
            ))
            return results
        
        if "description" not in problem:
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.WARNING,
                message="problem should have a description",
                field="problem.description",
            ))
        
        if "symptoms" in problem and not isinstance(problem["symptoms"], list):
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message="problem.symptoms must be a list",
                field="problem.symptoms",
            ))
        
        if "affected_apis" in problem and not isinstance(problem["affected_apis"], list):
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message="problem.affected_apis must be a list",
                field="problem.affected_apis",
            ))
        
        return results
    
    def _validate_solution(self, solution: Any) -> List[ValidationResult]:
        results = []
        
        if not isinstance(solution, dict):
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message="solution must be a dictionary",
                field="solution",
            ))
            return results
        
        if "description" not in solution:
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.WARNING,
                message="solution should have a description",
                field="solution.description",
            ))
        
        return results
    
    def _validate_metadata(self, metadata: Any) -> List[ValidationResult]:
        results = []
        
        if not isinstance(metadata, dict):
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message="metadata must be a dictionary",
                field="metadata",
            ))
            return results
        
        if "tags" in metadata and not isinstance(metadata["tags"], list):
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message="metadata.tags must be a list",
                field="metadata.tags",
            ))
        
        if "severity" in metadata and metadata["severity"] not in self.SEVERITY_LEVELS:
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.WARNING,
                message=f"Unknown severity: {metadata['severity']}",
                field="metadata.severity",
                suggestion=f"Use one of: {', '.join(self.SEVERITY_LEVELS)}",
            ))
        
        return results
    
    def validate_rule(self, rule_data: Dict[str, Any]) -> List[ValidationResult]:
        results = []
        
        required_fields = ["rule_id", "rule_type", "title", "description"]
        for field in required_fields:
            if field not in rule_data:
                results.append(ValidationResult(
                    valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Missing required field: {field}",
                    field=field,
                    suggestion=f"Add the '{field}' field",
                ))
        
        if "rule_id" in rule_data:
            id_result = self._validate_rule_id(rule_data["rule_id"])
            results.extend(id_result)
        
        if "rule_type" in rule_data and rule_data["rule_type"] not in self.RULE_TYPES:
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.WARNING,
                message=f"Unknown rule_type: {rule_data['rule_type']}",
                field="rule_type",
                suggestion=f"Use one of: {', '.join(self.RULE_TYPES)}",
            ))
        
        if "constraints" in rule_data and not isinstance(rule_data["constraints"], dict):
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message="constraints must be a dictionary",
                field="constraints",
            ))
        
        if "tags" in rule_data and not isinstance(rule_data["tags"], list):
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message="tags must be a list",
                field="tags",
            ))
        
        return results
    
    def _validate_rule_id(self, rule_id: str) -> List[ValidationResult]:
        results = []
        
        if not rule_id:
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message="rule_id cannot be empty",
                field="rule_id",
            ))
            return results
        
        if not re.match(r'^[a-z][a-z0-9_]*$', rule_id):
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message="rule_id must start with lowercase letter and contain only lowercase letters, numbers, and underscores",
                field="rule_id",
                suggestion="Use format: category_descriptive_name",
            ))
        
        if len(rule_id) > 100:
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.WARNING,
                message="rule_id is too long (max 100 characters)",
                field="rule_id",
            ))
        
        if rule_id in self.existing_ids:
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.WARNING,
                message="rule_id already exists",
                field="rule_id",
                suggestion="Use a unique rule_id",
            ))
        
        return results
    
    def validate_document(self, doc_content: str, doc_name: str = None) -> List[ValidationResult]:
        results = []
        
        if not doc_content or not doc_content.strip():
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message="Document content is empty",
            ))
            return results
        
        if not doc_content.startswith("#"):
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.INFO,
                message="Document should start with a title (# heading)",
                suggestion="Add a title at the beginning",
            ))
        
        lines = doc_content.split("\n")
        if len(lines) < 5:
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.INFO,
                message="Document seems too short",
                suggestion="Consider adding more content",
            ))
        
        if doc_name:
            if not re.match(r'^[a-z0-9_]+\.md$', doc_name):
                results.append(ValidationResult(
                    valid=False,
                    level=ValidationLevel.WARNING,
                    message="Document name should use lowercase letters, numbers, and underscores",
                    suggestion="Use format: descriptive_name.md",
                ))
        
        return results
    
    def check_duplicates(self, knowledge_data: Dict[str, Any], knowledge_type: str) -> List[ValidationResult]:
        results = []
        
        if not self.knowledge_base_path:
            return results
        
        if knowledge_type == "case":
            similar_cases = self._find_similar_cases(knowledge_data)
            for similar in similar_cases:
                results.append(ValidationResult(
                    valid=True,
                    level=ValidationLevel.WARNING,
                    message=f"Similar case exists: {similar['case_id']}",
                    suggestion="Consider updating existing case or ensure this is a different issue",
                ))
        
        return results
    
    def _find_similar_cases(self, case_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        similar = []
        
        if not self.knowledge_base_path:
            return similar
        
        cases_dir = self.knowledge_base_path / "cases"
        if not cases_dir.exists():
            return similar
        
        title = case_data.get("title", "").lower()
        problem_desc = case_data.get("problem", {}).get("description", "").lower()
        
        for case_file in cases_dir.glob("**/*.json"):
            try:
                with open(case_file, "r", encoding="utf-8") as f:
                    existing_case = json.load(f)
                
                existing_title = existing_case.get("title", "").lower()
                existing_desc = existing_case.get("problem", {}).get("description", "").lower()
                
                if self._calculate_similarity(title, existing_title) > 0.7:
                    similar.append(existing_case)
                elif self._calculate_similarity(problem_desc, existing_desc) > 0.7:
                    similar.append(existing_case)
            except Exception:
                pass
        
        return similar
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        infos = [r for r in results if r.level == ValidationLevel.INFO]
        
        return {
            "valid": len(errors) == 0,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "info_count": len(infos),
            "errors": [{"field": r.field, "message": r.message} for r in errors],
            "warnings": [{"field": r.field, "message": r.message} for r in warnings],
            "all_results": [r.to_dict() for r in results],
        }


def main():
    parser = argparse.ArgumentParser(description="Knowledge Validator for Triton Knowledge Base")
    parser.add_argument("--input", required=True, help="Path to knowledge file or JSON string")
    parser.add_argument("--type", required=True, choices=["case", "rule", "document"], help="Knowledge type")
    parser.add_argument("--knowledge-base", help="Path to knowledge base for duplicate checking")
    parser.add_argument("--check-duplicates", action="store_true", help="Check for duplicates")
    parser.add_argument("--doc-name", help="Document name (for document type)")
    
    args = parser.parse_args()
    
    validator = KnowledgeValidator(args.knowledge_base)
    
    input_path = Path(args.input)
    if input_path.exists():
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            data = json.loads(args.input)
        except json.JSONDecodeError:
            print(json.dumps({
                "valid": False,
                "errors": [{"message": "Invalid JSON input"}]
            }, indent=2))
            sys.exit(1)
    
    if args.type == "document":
        data = {"content": data.get("content", str(data)), "name": args.doc_name}
    
    result = validator.validate(data, args.type)
    
    if args.check_duplicates and args.knowledge_base:
        duplicate_results = validator.check_duplicates(data, args.type)
        if duplicate_results:
            result["duplicate_check"] = [r.to_dict() for r in duplicate_results]
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
