"""
Knowledge Indexer for Triton Knowledge Base

Provides indexing and retrieval functionality for the knowledge base.

Usage:
    python knowledge_indexer.py --knowledge-base <path> --action <action> [options]

Actions:
    --rebuild           Rebuild the entire index
    --search            Search knowledge by keywords/tags
    --get               Get a specific knowledge entry
    --increment-usage   Increment usage count for an entry
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


@dataclass
class KnowledgeEntry:
    id: str
    type: str
    subtype: str
    title: str
    content: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    created_at: str = ""
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "subtype": self.subtype,
            "title": self.title,
            "content": self.content,
            "tags": self.tags,
            "keywords": self.keywords,
            "created_at": self.created_at,
            "usage_count": self.usage_count,
        }


class KnowledgeIndexer:
    def __init__(self, knowledge_base_path: str):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.index: Dict[str, Any] = {
            "keywords": defaultdict(list),
            "tags": defaultdict(list),
            "types": defaultdict(list),
            "entries": {},
        }
        self._load_index()
    
    def _load_index(self):
        index_file = self.knowledge_base_path / "index.json"
        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.index["keywords"] = defaultdict(list, data.get("keywords", {}))
                self.index["tags"] = defaultdict(list, data.get("tags", {}))
                self.index["types"] = defaultdict(list, data.get("types", {}))
                self.index["entries"] = data.get("entries", {})
    
    def _save_index(self):
        index_file = self.knowledge_base_path / "index.json"
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        
        data = {
            "keywords": dict(self.index["keywords"]),
            "tags": dict(self.index["tags"]),
            "types": dict(self.index["types"]),
            "entries": self.index["entries"],
            "updated_at": datetime.now().isoformat(),
        }
        
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def index_entry(self, entry: KnowledgeEntry):
        entry_id = entry.id
        
        self.index["entries"][entry_id] = entry.to_dict()
        
        for keyword in entry.keywords:
            if entry_id not in self.index["keywords"][keyword]:
                self.index["keywords"][keyword].append(entry_id)
        
        for tag in entry.tags:
            if entry_id not in self.index["tags"][tag]:
                self.index["tags"][tag].append(entry_id)
        
        type_key = f"{entry.type}:{entry.subtype}"
        if entry_id not in self.index["types"][type_key]:
            self.index["types"][type_key].append(entry_id)
        
        self._save_index()
    
    def remove_entry(self, entry_id: str):
        if entry_id not in self.index["entries"]:
            return
        
        entry_data = self.index["entries"][entry_id]
        
        for keyword in entry_data.get("keywords", []):
            if entry_id in self.index["keywords"][keyword]:
                self.index["keywords"][keyword].remove(entry_id)
        
        for tag in entry_data.get("tags", []):
            if entry_id in self.index["tags"][tag]:
                self.index["tags"][tag].remove(entry_id)
        
        type_key = f"{entry_data['type']}:{entry_data['subtype']}"
        if entry_id in self.index["types"][type_key]:
            self.index["types"][type_key].remove(entry_id)
        
        del self.index["entries"][entry_id]
        self._save_index()
    
    def search(self, keywords: List[str] = None, tags: List[str] = None, 
               type_name: str = None, subtype: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        results = []
        seen_ids: Set[str] = set()
        
        keyword_matches = set()
        if keywords:
            for keyword in keywords:
                keyword_lower = keyword.lower()
                for indexed_keyword, entry_ids in self.index["keywords"].items():
                    if keyword_lower in indexed_keyword.lower():
                        keyword_matches.update(entry_ids)
        
        tag_matches = set()
        if tags:
            for tag in tags:
                tag_lower = tag.lower()
                for indexed_tag, entry_ids in self.index["tags"].items():
                    if tag_lower == indexed_tag.lower():
                        tag_matches.update(entry_ids)
        
        type_matches = set()
        if type_name:
            if subtype:
                type_key = f"{type_name}:{subtype}"
                type_matches.update(self.index["types"].get(type_key, []))
            else:
                for type_key, entry_ids in self.index["types"].items():
                    if type_key.startswith(f"{type_name}:"):
                        type_matches.update(entry_ids)
        
        candidate_ids = None
        if keywords or tags or type_name:
            if keywords and not keyword_matches:
                return []
            if tags and not tag_matches:
                return []
            if type_name and not type_matches:
                return []
            
            candidate_ids = keyword_matches if keywords else None
            if tags:
                candidate_ids = tag_matches if candidate_ids is None else candidate_ids & tag_matches
            if type_name:
                candidate_ids = type_matches if candidate_ids is None else candidate_ids & type_matches
        else:
            candidate_ids = set(self.index["entries"].keys())
        
        if candidate_ids is None:
            candidate_ids = set(self.index["entries"].keys())
        
        for entry_id in candidate_ids:
            if entry_id in self.index["entries"]:
                entry = self.index["entries"][entry_id]
                results.append({
                    "id": entry["id"],
                    "type": entry["type"],
                    "subtype": entry["subtype"],
                    "title": entry["title"],
                    "relevance": self._calculate_relevance(entry, keywords, tags),
                    "summary": self._generate_summary(entry),
                    "tags": entry.get("tags", []),
                })
        
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:limit]
    
    def _calculate_relevance(self, entry: Dict, keywords: List[str], tags: List[str]) -> float:
        score = 0.5
        
        if keywords:
            entry_keywords = set(k.lower() for k in entry.get("keywords", []))
            keyword_matches = sum(1 for k in keywords if k.lower() in entry_keywords)
            score += min(keyword_matches / len(keywords), 0.3)
        
        if tags:
            entry_tags = set(t.lower() for t in entry.get("tags", []))
            tag_matches = sum(1 for t in tags if t.lower() in entry_tags)
            score += min(tag_matches / len(tags), 0.2)
        
        return min(score, 1.0)
    
    def _generate_summary(self, entry: Dict) -> str:
        content = entry.get("content", {})
        
        if entry["type"] == "case":
            problem = content.get("problem", {})
            desc = problem.get("description", "")
            return desc[:100] + "..." if len(desc) > 100 else desc
        elif entry["type"] == "rule":
            desc = content.get("description", "")
            return desc[:100] + "..." if len(desc) > 100 else desc
        else:
            return entry.get("title", "")
    
    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        return self.index["entries"].get(entry_id)
    
    def get_full_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        entry_summary = self.index["entries"].get(entry_id)
        if not entry_summary:
            return None
        
        entry_type = entry_summary["type"]
        
        if entry_type == "case":
            cases_dir = self.knowledge_base_path / "cases"
            for case_file in cases_dir.glob("**/*.json"):
                try:
                    with open(case_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("case_id") == entry_id:
                        return data
                except Exception:
                    pass
        elif entry_type == "rule":
            rules_dir = self.knowledge_base_path / "rules"
            for rule_file in rules_dir.glob("*.json"):
                try:
                    with open(rule_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("rule_id") == entry_id:
                        return data
                except Exception:
                    pass
        
        return entry_summary
    
    def increment_usage(self, entry_id: str):
        if entry_id in self.index["entries"]:
            self.index["entries"][entry_id]["usage_count"] += 1
            self._save_index()
            return True
        return False
    
    def get_popular_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        entries = list(self.index["entries"].values())
        entries.sort(key=lambda x: x.get("usage_count", 0), reverse=True)
        return entries[:limit]
    
    def get_promotion_candidates(self, threshold: int = 5) -> List[Dict[str, Any]]:
        candidates = []
        for entry in self.index["entries"].values():
            if entry.get("usage_count", 0) >= threshold:
                candidates.append({
                    "knowledge_id": entry["id"],
                    "type": entry["type"],
                    "subtype": entry["subtype"],
                    "title": entry["title"],
                    "usage_count": entry["usage_count"],
                    "suggested_target": self._suggest_target(entry),
                    "priority": "high" if entry["usage_count"] >= 10 else "medium",
                })
        candidates.sort(key=lambda x: x["usage_count"], reverse=True)
        return candidates
    
    def _suggest_target(self, entry: Dict) -> str:
        if entry["type"] == "case":
            if entry["subtype"] in ["precision_issue", "compilation_error", "runtime_error"]:
                return "triton-code-generator/references/"
            elif entry["subtype"] == "optimization":
                return "triton-performance-optimizer/references/"
            elif entry["subtype"] == "conversion":
                return "cuda-to-ascend-converter/references/"
        elif entry["type"] == "rule":
            return "triton-code-generator/references/"
        return "triton-code-generator/references/"
    
    def rebuild_index(self):
        self.index = {
            "keywords": defaultdict(list),
            "tags": defaultdict(list),
            "types": defaultdict(list),
            "entries": {},
        }
        
        cases_dir = self.knowledge_base_path / "cases"
        if cases_dir.exists():
            for case_file in cases_dir.glob("**/*.json"):
                try:
                    with open(case_file, "r", encoding="utf-8") as f:
                        case_data = json.load(f)
                    
                    entry = self._parse_case(case_data)
                    if entry:
                        self.index_entry(entry)
                except Exception as e:
                    print(f"Error parsing {case_file}: {e}", file=sys.stderr)
        
        rules_dir = self.knowledge_base_path / "rules"
        if rules_dir.exists():
            for rule_file in rules_dir.glob("*.json"):
                try:
                    with open(rule_file, "r", encoding="utf-8") as f:
                        rule_data = json.load(f)
                    
                    entry = self._parse_rule(rule_data)
                    if entry:
                        self.index_entry(entry)
                except Exception as e:
                    print(f"Error parsing {rule_file}: {e}", file=sys.stderr)
        
        self._save_index()
        return {"success": True, "entries_indexed": len(self.index["entries"])}
    
    def _parse_case(self, case_data: Dict[str, Any]) -> Optional[KnowledgeEntry]:
        try:
            case_id = case_data.get("case_id", "")
            case_type = case_data.get("case_type", "")
            
            keywords = self._extract_keywords(case_data)
            
            return KnowledgeEntry(
                id=case_id,
                type="case",
                subtype=case_type,
                title=case_data.get("title", ""),
                content=case_data,
                tags=case_data.get("metadata", {}).get("tags", []),
                keywords=keywords,
                created_at=case_data.get("metadata", {}).get("created_at", ""),
                usage_count=case_data.get("metadata", {}).get("usage_count", 0),
            )
        except Exception:
            return None
    
    def _parse_rule(self, rule_data: Dict[str, Any]) -> Optional[KnowledgeEntry]:
        try:
            rule_id = rule_data.get("rule_id", "")
            rule_type = rule_data.get("rule_type", "")
            
            keywords = self._extract_keywords(rule_data)
            
            return KnowledgeEntry(
                id=rule_id,
                type="rule",
                subtype=rule_type,
                title=rule_data.get("title", ""),
                content=rule_data,
                tags=rule_data.get("tags", []),
                keywords=keywords,
                created_at=rule_data.get("created_at", ""),
                usage_count=0,
            )
        except Exception:
            return None
    
    def _extract_keywords(self, data: Dict[str, Any]) -> List[str]:
        keywords = set()
        
        text_fields = ["title", "description", "problem", "solution"]
        for field in text_fields:
            value = data.get(field)
            if isinstance(value, str):
                words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', value.lower())
                keywords.update(words)
            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, str):
                        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', v.lower())
                        keywords.update(words)
        
        return list(keywords)


def main():
    parser = argparse.ArgumentParser(description="Knowledge Indexer for Triton Knowledge Base")
    parser.add_argument("--knowledge-base", required=True, help="Path to knowledge base directory")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the entire index")
    parser.add_argument("--search", action="store_true", help="Search knowledge")
    parser.add_argument("--get", metavar="ID", help="Get a specific knowledge entry")
    parser.add_argument("--increment-usage", metavar="ID", help="Increment usage count")
    parser.add_argument("--keywords", nargs="+", help="Search keywords")
    parser.add_argument("--tags", nargs="+", help="Search tags")
    parser.add_argument("--type", dest="type_name", help="Knowledge type (case/rule)")
    parser.add_argument("--subtype", help="Knowledge subtype")
    parser.add_argument("--limit", type=int, default=10, help="Maximum results")
    parser.add_argument("--promotion-candidates", action="store_true", help="Get promotion candidates")
    
    args = parser.parse_args()
    
    indexer = KnowledgeIndexer(args.knowledge_base)
    
    if args.rebuild:
        result = indexer.rebuild_index()
        print(json.dumps(result, indent=2))
    
    elif args.search:
        results = indexer.search(
            keywords=args.keywords,
            tags=args.tags,
            type_name=args.type_name,
            subtype=args.subtype,
            limit=args.limit
        )
        print(json.dumps({
            "success": True,
            "action": "search",
            "results": results,
            "total": len(results)
        }, indent=2, ensure_ascii=False))
    
    elif args.get:
        entry = indexer.get_full_entry(args.get)
        if entry:
            print(json.dumps({
                "success": True,
                "action": "get",
                "result": entry
            }, indent=2, ensure_ascii=False))
        else:
            print(json.dumps({
                "success": False,
                "error": f"Entry not found: {args.get}"
            }, indent=2))
    
    elif args.increment_usage:
        success = indexer.increment_usage(args.increment_usage)
        print(json.dumps({
            "success": success,
            "action": "increment_usage",
            "entry_id": args.increment_usage
        }, indent=2))
    
    elif args.promotion_candidates:
        candidates = indexer.get_promotion_candidates()
        print(json.dumps({
            "success": True,
            "action": "promotion_candidates",
            "promotion_suggestions": candidates
        }, indent=2, ensure_ascii=False))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
