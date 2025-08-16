"""
Security Agent for static code analysis using LangGraph
"""
import os
import re
import json
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pygments.lexers
from pygments.util import ClassNotFound
from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SecurityState:
    """Security analysis state"""
    input: Dict[str, Any]
    language: Optional[str] = None
    risk_score: float = 0.0
    threshold: float = 60.0
    findings: List[Dict[str, Any]] = None
    proposed_fix: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.findings is None:
            self.findings = []


class SecurityModel:
    """CodeT5 security classifier wrapper"""
    
    def __init__(self, model_path: str = "codet5-security-classifier/"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = self._get_device()
        self._load_model()
    
    def _get_device(self) -> str:
        """Auto-detect device"""
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"
    
    def _load_model(self):
        """Load model and tokenizer"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def classify_security(self, code: str, max_length: int = 1024) -> Dict[str, Any]:
        """Classify code security vulnerability"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                code,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            # Assume binary classification: [safe, vulnerable]
            prob_vuln = probs[0][1].item()
            label = "vulnerable" if prob_vuln > 0.5 else "safe"
            
            return {
                "prob_vuln": prob_vuln,
                "logits": logits.cpu().numpy().tolist(),
                "label": label
            }
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {"prob_vuln": 0.0, "logits": [], "label": "error"}


class SecurityAgent:
    """Main security agent with LangGraph pipeline"""
    
    def __init__(self, model_path: str = "codet5-security-classifier/"):
        self.model_path = model_path
        self.security_model = None
        self._init_model()
        self.graph = self._build_graph()
    
    def _init_model(self):
        """Initialize security model if available"""
        try:
            if os.path.exists(self.model_path):
                self.security_model = SecurityModel(self.model_path)
            else:
                logger.warning(f"Model not found at {self.model_path}, using heuristics only")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph pipeline"""
        graph = StateGraph(SecurityState)
        
        # Add nodes
        graph.add_node("detect_language", self._detect_language_node)
        graph.add_node("branch", self._branch_node)
        graph.add_node("classify_c", self._classify_c_node)
        graph.add_node("non_c_assess", self._non_c_assess_node)
        graph.add_node("propose_fix", self._propose_fix_node)
        graph.add_node("aggregate", self._aggregate_node)
        
        # Add edges
        graph.add_edge("detect_language", "branch")
        graph.add_conditional_edges(
            "branch",
            self._route_by_language,
            {"c": "classify_c", "non_c": "non_c_assess"}
        )
        graph.add_edge("classify_c", "propose_fix")
        graph.add_edge("non_c_assess", "propose_fix")
        graph.add_edge("propose_fix", "aggregate")
        
        graph.set_entry_point("detect_language")
        graph.set_finish_point("aggregate")
        
        return graph.compile()
    
    def _detect_language_node(self, state: SecurityState) -> SecurityState:
        """Detect programming language"""
        language = state.input.get("language")
        
        if not language:
            code = state.input.get("code", "")
            language = self._detect_language(code)
        
        state.language = language.lower()
        logger.info(f"Detected language: {state.language}")
        return state
    
    def _detect_language(self, code: str) -> str:
        """Simple language detection using pygments"""
        try:
            lexer = pygments.lexers.guess_lexer(code)
            return lexer.name.lower()
        except ClassNotFound:
            # Simple heuristics
            if "#include" in code or "int main" in code:
                return "c"
            elif "def " in code or "import " in code:
                return "python"
            elif "function " in code or "const " in code:
                return "javascript"
            return "unknown"
    
    def _branch_node(self, state: SecurityState) -> SecurityState:
        """Branch based on language"""
        # No processing needed, just pass through
        return state
    
    def _route_by_language(self, state: SecurityState) -> str:
        """Route to appropriate analysis node"""
        if state.language in ["c", "cpp", "c++"]:
            return "c"
        return "non_c"
    
    def _classify_c_node(self, state: SecurityState) -> SecurityState:
        """Classify C code using model"""
        code = state.input.get("code", "")
        
        if self.security_model:
            result = self.security_model.classify_security(code)
            state.risk_score = round(result["prob_vuln"] * 100, 2)
            state.findings.append({
                "rule": "model:codet5-security-classifier",
                "detail": f"Model prediction: {result['label']} (confidence: {result['prob_vuln']:.3f})",
                "line": None
            })
        else:
            # Fallback to heuristics
            state = self._analyze_c_heuristics(state)
        
        # Add pattern-based findings
        state = self._analyze_c_patterns(state)
        
        logger.info(f"C analysis complete. Risk score: {state.risk_score}%")
        return state
    
    def _analyze_c_heuristics(self, state: SecurityState) -> SecurityState:
        """Heuristic analysis for C code"""
        code = state.input.get("code", "")
        risk_factors = 0
        
        # Dangerous functions
        dangerous_funcs = ["gets"]  # Remove strcpy, strcat, sprintf from here
        for func in dangerous_funcs:
            pattern = rf'\b{func}\s*\('
            if re.search(pattern, code):
                risk_factors += 25
                state.findings.append({
                    "rule": f"dangerous_function:{func}",
                    "detail": f"Unsafe function '{func}' detected",
                    "line": None
                })
        
        # Check for strcpy, strcat, sprintf without safe versions
        if re.search(r'\bstrcpy\s*\(', code) and not re.search(r'\bstrncpy\s*\(', code):
            risk_factors += 25
            state.findings.append({
                "rule": "dangerous_function:strcpy",
                "detail": "Unsafe function 'strcpy' detected",
                "line": None
            })
            
        if re.search(r'\bstrcat\s*\(', code) and not re.search(r'\bstrncat\s*\(', code):
            risk_factors += 25
            state.findings.append({
                "rule": "dangerous_function:strcat", 
                "detail": "Unsafe function 'strcat' detected",
                "line": None
            })
            
        if re.search(r'\bsprintf\s*\(', code) and not re.search(r'\bsnprintf\s*\(', code):
            risk_factors += 25
            state.findings.append({
                "rule": "dangerous_function:sprintf",
                "detail": "Unsafe function 'sprintf' detected", 
                "line": None
            })
        
        # Buffer operations without bounds checking
        if re.search(r'\w+\[\w*\]', code) and "sizeof" not in code:
            risk_factors += 15
            state.findings.append({
                "rule": "unbounded_array",
                "detail": "Array access without bounds checking",
                "line": None
            })
        
        state.risk_score = min(risk_factors, 100)
        return state
    
    def _analyze_c_patterns(self, state: SecurityState) -> SecurityState:
        """Pattern-based C analysis"""
        code = state.input.get("code", "")
        
        # Specific vulnerability patterns
        patterns = [
            (r'\bgets\s*\(', "CWE-120: Buffer overflow via gets()"),
            (r'\bstrcpy\s*\([^,]+,\s*[^)]+\)', "CWE-120: Unsafe string copy"),
            (r'\bsprintf\s*\([^,]+,', "CWE-134: Uncontrolled format string"),
            (r'\*\s*\w+\s*=', "Potential null pointer dereference"),
        ]
        
        for pattern, desc in patterns:
            if re.search(pattern, code):
                state.findings.append({
                    "rule": f"pattern:{pattern}",
                    "detail": desc,
                    "line": None
                })
                state.risk_score = min(state.risk_score + 10, 100)
        
        return state
    
    def _non_c_assess_node(self, state: SecurityState) -> SecurityState:
        """Assess non-C languages using heuristics"""
        code = state.input.get("code", "")
        language = state.language
        
        risk_score = 0
        
        if language == "python":
            risk_score = self._assess_python(code, state)
        elif language in ["javascript", "js"]:
            risk_score = self._assess_javascript(code, state)
        else:
            risk_score = self._assess_generic(code, state)
        
        state.risk_score = risk_score
        logger.info(f"{language} analysis complete. Risk score: {state.risk_score}%")
        return state
    
    def _assess_python(self, code: str, state: SecurityState) -> float:
        """Python-specific security assessment"""
        risk_score = 0
        
        # Dangerous patterns
        patterns = [
            (r'eval\s*\(', "Code injection via eval()", 30),
            (r'exec\s*\(', "Code injection via exec()", 30),
            (r'input\s*\(\)\s*.*password', "Password in plain input", 20),
            (r'pickle\.loads?\s*\(', "Unsafe deserialization", 25),
            (r'subprocess\.(call|run|Popen)', "Command injection risk", 15),
        ]
        
        for pattern, desc, score in patterns:
            if re.search(pattern, code, re.IGNORECASE):
                risk_score += score
                state.findings.append({
                    "rule": f"python:{pattern}",
                    "detail": desc,
                    "line": None
                })
        
        return min(risk_score, 100)
    
    def _assess_javascript(self, code: str, state: SecurityState) -> float:
        """JavaScript-specific security assessment"""
        risk_score = 0
        
        patterns = [
            (r'eval\s*\(', "Code injection via eval()", 30),
            (r'innerHTML\s*=', "XSS via innerHTML", 25),
            (r'document\.write\s*\(', "XSS via document.write", 20),
            (r'setTimeout\s*\(\s*["\']', "Code injection in setTimeout", 25),
        ]
        
        for pattern, desc, score in patterns:
            if re.search(pattern, code, re.IGNORECASE):
                risk_score += score
                state.findings.append({
                    "rule": f"javascript:{pattern}",
                    "detail": desc,
                    "line": None
                })
        
        return min(risk_score, 100)
    
    def _assess_generic(self, code: str, state: SecurityState) -> float:
        """Generic security assessment"""
        risk_score = 0
        
        # Basic patterns
        if "password" in code.lower() and "=" in code:
            risk_score += 10
            state.findings.append({
                "rule": "generic:hardcoded_password",
                "detail": "Potential hardcoded password",
                "line": None
            })
        
        return min(risk_score, 100)
    
    def _propose_fix_node(self, state: SecurityState) -> SecurityState:
        """Propose security fixes"""
        threshold = state.input.get("metadata", {}).get("threshold", 0.6) * 100
        state.threshold = threshold
        
        if state.risk_score >= threshold:
            state.proposed_fix = self._generate_fix_proposal(state)
        
        return state
    
    def _generate_fix_proposal(self, state: SecurityState) -> Dict[str, Any]:
        """Generate fix proposals based on findings"""
        language = state.language
        code = state.input.get("code", "")
        
        if language in ["c", "cpp"]:
            return self._generate_c_fix(code, state.findings)
        elif language == "python":
            return self._generate_python_fix(code, state.findings)
        else:
            return self._generate_generic_fix(code, state.findings)
    
    def _generate_c_fix(self, code: str, findings: List[Dict]) -> Dict[str, Any]:
        """Generate C-specific fixes"""
        fixed_code = code
        notes = []
        
        # Replace dangerous functions
        replacements = {
            "gets(": "fgets(",
            "strcpy(": "strncpy(",
            "strcat(": "strncat(",
            "sprintf(": "snprintf("
        }
        
        for unsafe, safe in replacements.items():
            if unsafe in fixed_code:
                fixed_code = fixed_code.replace(unsafe, safe)
                notes.append(f"Replaced {unsafe} with {safe}")
        
        # Generate diff
        patch = self._generate_diff(code, fixed_code)
        
        return {
            "strategy": "Replace unsafe functions with bounds-checked alternatives",
            "patch": patch,
            "notes": "; ".join(notes) if notes else "Add bounds checking and input validation"
        }
    
    def _generate_python_fix(self, code: str, findings: List[Dict]) -> Dict[str, Any]:
        """Generate Python-specific fixes"""
        suggestions = []
        
        for finding in findings:
            if "eval" in finding["rule"]:
                suggestions.append("Avoid eval(); use ast.literal_eval() for safe evaluation")
            elif "pickle" in finding["rule"]:
                suggestions.append("Use JSON instead of pickle for data serialization")
            elif "subprocess" in finding["rule"]:
                suggestions.append("Sanitize inputs and use shell=False in subprocess calls")
        
        return {
            "strategy": "Apply Python security best practices",
            "patch": "# Apply the following security improvements:\n" + "\n".join(f"# - {s}" for s in suggestions),
            "notes": "Review input validation and sanitization"
        }
    
    def _generate_generic_fix(self, code: str, findings: List[Dict]) -> Dict[str, Any]:
        """Generate generic fixes"""
        return {
            "strategy": "General security improvements",
            "patch": "# Review code for security vulnerabilities\n# Apply language-specific security best practices",
            "notes": "Consider using security linters and static analysis tools"
        }
    
    def _generate_diff(self, original: str, fixed: str) -> str:
        """Generate simple diff"""
        if original == fixed:
            return "No changes needed"
        
        return f"--- original\n+++ fixed\n@@ -1,{len(original.splitlines())} +1,{len(fixed.splitlines())} @@\n" + \
               "\n".join(f"-{line}" for line in original.splitlines()) + "\n" + \
               "\n".join(f"+{line}" for line in fixed.splitlines())
    
    def _aggregate_node(self, state: SecurityState) -> SecurityState:
        """Final aggregation"""
        return state
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main analysis entry point"""
        try:
            # Initialize state
            state = SecurityState(input=input_data)
            
            # Set threshold
            threshold = input_data.get("metadata", {}).get("threshold", 0.6)
            state.threshold = threshold * 100
            
            # Run analysis manually without LangGraph for now
            # Language detection
            language = input_data.get("language")
            if not language:
                code = input_data.get("code", "")
                language = self._detect_language(code)
            state.language = language.lower()
            
            # Branch based on language
            if state.language in ["c", "cpp", "c++"]:
                # C analysis
                code = input_data.get("code", "")
                
                if self.security_model:
                    result = self.security_model.classify_security(code)
                    state.risk_score = round(result["prob_vuln"] * 100, 2)
                    state.findings.append({
                        "rule": "model:codet5-security-classifier",
                        "detail": f"Model prediction: {result['label']} (confidence: {result['prob_vuln']:.3f})",
                        "line": None
                    })
                else:
                    # Fallback to heuristics
                    state = self._analyze_c_heuristics(state)
                
                # Add pattern-based findings
                state = self._analyze_c_patterns(state)
            else:
                # Non-C analysis
                code = input_data.get("code", "")
                
                if state.language == "python":
                    risk_score = self._assess_python(code, state)
                elif state.language in ["javascript", "js"]:
                    risk_score = self._assess_javascript(code, state)
                else:
                    risk_score = self._assess_generic(code, state)
                
                state.risk_score = risk_score
            
            # Propose fixes if needed
            if state.risk_score >= state.threshold:
                state.proposed_fix = self._generate_fix_proposal(state)
            
            # Build response
            return {
                "language": state.language,
                "risk_score": state.risk_score,
                "is_vulnerable": state.risk_score >= state.threshold,
                "threshold": state.threshold,
                "findings": state.findings,
                "proposed_fix": state.proposed_fix
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                "language": "unknown",
                "risk_score": 0.0,
                "is_vulnerable": False,
                "threshold": 60.0,
                "findings": [{"rule": "error", "detail": str(e), "line": None}],
                "proposed_fix": None
            }


def run_security_agent(code: str, language: str = None, threshold: float = None) -> Dict[str, Any]:
    """Module API for security analysis"""
    input_data = {
        "code": code,
        "language": language,
        "metadata": {"threshold": threshold or 0.6}
    }
    
    agent = SecurityAgent()
    return agent.analyze(input_data)


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Security static analysis tool")
    parser.add_argument("--path", help="Path to source file")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    parser.add_argument("--threshold", type=float, default=0.6, help="Vulnerability threshold (0-1)")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON")
    parser.add_argument("--language", help="Override language detection")
    
    args = parser.parse_args()
    
    # Read code
    if args.stdin:
        code = sys.stdin.read()
        path = None
    elif args.path:
        if not os.path.exists(args.path):
            print(f"Error: File {args.path} not found", file=sys.stderr)
            sys.exit(2)
        with open(args.path, 'r', encoding='utf-8') as f:
            code = f.read()
        path = args.path
    else:
        parser.print_help()
        sys.exit(2)
    
    # Prepare input
    input_data = {
        "code": code,
        "language": args.language,
        "metadata": {
            "path": path,
            "threshold": args.threshold
        }
    }
    
    # Run analysis
    agent = SecurityAgent()
    result = agent.analyze(input_data)
    
    # Output result
    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))
    
    # Exit code
    sys.exit(1 if result["is_vulnerable"] else 0)
