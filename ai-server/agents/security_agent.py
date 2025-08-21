"""
Hybrid Security Agent combining rule-based scanning with LLM intelligence
"""
import os
import re
import json
import torch
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pygments.lexers
from pygments.util import ClassNotFound
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisDepth(Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class RiskLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityFinding:
    """단일 보안 발견사항"""
    rule_id: str
    title: str
    description: str
    risk_level: RiskLevel
    confidence: float
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    cwe_id: Optional[str] = None
    fix_suggestion: Optional[str] = None


@dataclass
class SecurityState:
    """Hybrid Security analysis state"""
    input: Dict[str, Any]
    language: Optional[str] = None
    code: str = ""
    analysis_depth: AnalysisDepth = AnalysisDepth.STANDARD
    
    # Rule-based 결과
    rule_based_findings: List[SecurityFinding] = None
    rule_based_score: float = 0.0
    
    # LLM 분석 결과
    llm_findings: List[SecurityFinding] = None
    llm_analysis: Optional[str] = None
    needs_llm_analysis: bool = False
    
    # 최종 결과
    final_findings: List[SecurityFinding] = None
    risk_score: float = 0.0
    threshold: float = 60.0
    proposed_fix: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.rule_based_findings is None:
            self.rule_based_findings = []
        if self.llm_findings is None:
            self.llm_findings = []
        if self.final_findings is None:
            self.final_findings = []
        self.code = self.input.get("code", "")


class VulnerabilityPatterns:
    """취약점 패턴 데이터베이스"""
    
    # Critical patterns that always trigger LLM analysis
    CRITICAL_PATTERNS = {
        "sql_injection": [
            r"(SELECT|INSERT|UPDATE|DELETE).*\+.*[\'\"]",
            r"query.*=.*[\'\"].*\+",
            r"execute\s*\(\s*[\"'].*\+",
        ],
        "command_injection": [
            r"system\s*\(\s*.*\+",
            r"exec\s*\(\s*.*\+",
            r"eval\s*\(\s*.*\+",
        ],
        "xss": [
            r"innerHTML\s*=.*\+",
            r"document\.write\s*\(.*\+",
            r"\.html\s*\(.*\+",
        ]
    }
    
    # Language-specific patterns
    LANGUAGE_PATTERNS = {
        "c": {
            "buffer_overflow": [r"gets\s*\(", r"strcpy\s*\(", r"sprintf\s*\("],
            "format_string": [r"printf\s*\([^,)]+\)"],
            "memory_issues": [r"malloc.*free", r"double.*free"],
        },
        "javascript": {
            "prototype_pollution": [r"__proto__", r"constructor\.prototype"],
            "weak_crypto": [r"Math\.random\(\)", r"btoa\(", r"atob\("],
            "insecure_cors": [r"Access-Control-Allow-Origin.*\*"],
        },
        "python": {
            "sql_injection": [
                r"\.execute\s*\(\s*[\"'].*\+",
                r"SELECT.*FROM.*WHERE.*[\"'].*\+.*[\"']",
                r"query.*=.*[\"'].*\+.*[\"']"
            ],
            "command_injection": [
                r"subprocess\.call\s*\(.*shell\s*=\s*True",
                r"os\.system\s*\(.*\+",
                r"subprocess\.run\s*\(.*shell\s*=\s*True"
            ],
            "hardcoded_secrets": [
                r"password\s*=\s*[\"'][^\"']{3,}[\"']",
                r"api_key\s*=\s*[\"'][^\"']{10,}[\"']",
                r"secret\s*=\s*[\"'][^\"']{5,}[\"']"
            ],
            "weak_random": [
                r"random\.randint\s*\(\s*\d+\s*,\s*\d+\s*\)",
                r"random\.choice\s*\(",
                r"random\.random\s*\("
            ],
            "path_traversal": [
                r"open\s*\(\s*[\"'][^\"']*\+",
                r"file_path.*=.*[\"'][^\"']*\+.*[\"']"
            ],
            "pickle_injection": [r"pickle\.loads?", r"cPickle\.loads?"],
            "yaml_injection": [r"yaml\.load\(", r"yaml\.unsafe_load"],
            "code_injection": [r"exec\s*\(", r"eval\s*\(", r"compile\s*\("],
        }
    }


class RuleBasedScanner:
    """빠른 규칙 기반 스캐너"""
    
    def __init__(self):
        self.patterns = VulnerabilityPatterns()
    
    def quick_scan(self, code: str, language: str) -> Tuple[List[SecurityFinding], float, bool]:
        """빠른 스캔 (동기식)"""
        return self.scan(code, language)
    
    def scan(self, code: str, language: str) -> Tuple[List[SecurityFinding], float, bool]:
        """
        규칙 기반 스캔 수행
        Returns: (findings, risk_score, needs_llm_analysis)
        """
        findings = []
        risk_score = 0.0
        needs_llm = False
        detected_vulns = set()  # 중복 방지
        
        # Critical patterns check
        for vuln_type, patterns in self.patterns.CRITICAL_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    if vuln_type not in detected_vulns:
                        finding = self._create_detailed_finding(vuln_type, match, code, "critical")
                        findings.append(finding)
                        risk_score += 30
                        needs_llm = True
                        detected_vulns.add(vuln_type)
                        break  # 같은 타입은 한 번만 추가
        
        # Language-specific patterns
        lang_patterns = self.patterns.LANGUAGE_PATTERNS.get(language.lower(), {})
        for vuln_type, patterns in lang_patterns.items():
            if vuln_type in detected_vulns:
                continue  # 이미 감지된 취약점은 스킵
                
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    risk_level = self._determine_risk_level(vuln_type)
                    finding = self._create_detailed_finding(vuln_type, match, code, language)
                    findings.append(finding)
                    risk_score += self._get_risk_score(risk_level)
                    detected_vulns.add(vuln_type)
                    
                    if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                        needs_llm = True
                    break  # 같은 타입은 한 번만 추가
        
        # Generic security anti-patterns
        generic_findings, generic_score = self._scan_generic_patterns(code, detected_vulns)
        findings.extend(generic_findings)
        risk_score += generic_score
        
        return findings, min(risk_score, 100), needs_llm
    
    def _create_detailed_finding(self, vuln_type: str, match, code: str, context: str) -> SecurityFinding:
        """구체적인 발견사항 생성"""
        
        # 취약점 유형별 상세 설명
        vuln_descriptions = {
            "sql_injection": "SQL 쿼리에 사용자 입력이 직접 연결되어 데이터베이스 조작이 가능합니다",
            "command_injection": "시스템 명령에 사용자 입력이 포함되어 임의 명령 실행이 가능합니다",
            "xss": "사용자 입력이 HTML에 직접 출력되어 스크립트 인젝션이 가능합니다",
            "buffer_overflow": "버퍼 크기 검증 없이 데이터를 복사하여 메모리 오버플로우가 발생할 수 있습니다",
            "format_string": "포맷 스트링에 사용자 입력이 포함되어 메모리 읽기/쓰기가 가능합니다",
            "memory_issues": "메모리 할당/해제 과정에서 메모리 누수나 이중 해제가 발생할 수 있습니다",
            "hardcoded_secrets": "소스코드에 하드코딩된 비밀번호나 API 키가 노출되어 있습니다",
            "weak_random": "예측 가능한 난수 생성으로 인해 보안 토큰이 추측될 수 있습니다",
            "path_traversal": "파일 경로에 사용자 입력이 포함되어 시스템 파일 접근이 가능합니다",
            "insecure_cookie": "쿠키 보안 설정이 부적절하여 세션 하이재킹이 가능합니다",
            "weak_crypto": "약한 암호화 알고리즘이나 키를 사용하여 데이터 복호화가 쉽습니다"
        }
        
        # 매칭된 코드 부분 추출 (간단히)
        matched_text = match.group() if hasattr(match, 'group') else str(match)[:50]
        
        description = vuln_descriptions.get(vuln_type, f"{vuln_type} 취약점이 감지되었습니다")
        risk_level = self._determine_risk_level(vuln_type)
        
        return SecurityFinding(
            rule_id=f"{context}_{vuln_type}",
            title=f"{vuln_type.replace('_', ' ').title()}",
            description=f"{description} (발견: {matched_text})",
            risk_level=risk_level,
            confidence=0.8 if context == "critical" else 0.7,
            code_snippet=matched_text
        )
        
        # Generic security anti-patterns
        generic_findings, generic_score = self._scan_generic_patterns(code, detected_vulns)
        findings.extend(generic_findings)
        risk_score += generic_score
        
        # 취약성 여부 판단: 중간 이상 위험도 문제가 있거나 위험도가 30% 이상일 때
        is_vulnerable = any(f.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] for f in findings) or risk_score >= 30
        
        return findings, min(risk_score, 100), needs_llm
    
    def _determine_risk_level(self, vuln_type: str) -> RiskLevel:
        """취약점 유형에 따른 위험도 결정"""
        high_risk = ["buffer_overflow", "command_injection", "sql_injection"]
        medium_risk = ["xss", "weak_crypto", "format_string"]
        
        if vuln_type in high_risk:
            return RiskLevel.HIGH
        elif vuln_type in medium_risk:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
    
    def _get_risk_score(self, risk_level: RiskLevel) -> float:
        """위험도별 점수"""
        scores = {
            RiskLevel.CRITICAL: 25,
            RiskLevel.HIGH: 20,
            RiskLevel.MEDIUM: 15,
            RiskLevel.LOW: 5,
            RiskLevel.INFO: 1
        }
        return scores.get(risk_level, 0)
    
    def _scan_generic_patterns(self, code: str, detected_vulns: set) -> Tuple[List[SecurityFinding], float]:
        """일반적인 보안 패턴 스캔"""
        findings = []
        score = 0.0
        
        # Hardcoded secrets (이미 감지되지 않았다면)
        if "hardcoded_secrets" not in detected_vulns:
            secret_patterns = [
                (r"password\s*=\s*[\"'][^\"']{3,}[\"']", "Hardcoded Password", "하드코딩된 비밀번호: 환경변수나 설정 파일을 사용하세요"),
                (r"api[_-]?key\s*=\s*[\"'][^\"']{10,}[\"']", "Hardcoded API Key", "하드코딩된 API 키: .env 파일이나 보안 저장소를 사용하세요"),
                (r"secret\s*=\s*[\"'][^\"']{5,}[\"']", "Hardcoded Secret", "하드코딩된 시크릿: 암호화된 설정 관리 도구를 사용하세요"),
                (r"jwt[_-]?secret\s*=\s*[\"'][^\"']{5,}[\"']", "Hardcoded JWT Secret", "하드코딩된 JWT 시크릿: 환경변수로 관리하세요"),
            ]
            
            for pattern, title, desc in secret_patterns:
                match = re.search(pattern, code, re.IGNORECASE)
                if match:
                    finding = SecurityFinding(
                        rule_id=f"hardcoded_{title.lower().replace(' ', '_')}",
                        title=title,
                        description=f"{desc} (발견: {match.group()[:30]}...)",
                        risk_level=RiskLevel.MEDIUM,
                        confidence=0.8,
                        code_snippet=match.group()
                    )
                    findings.append(finding)
                    score += 15
                    # 하나라도 찾으면 hardcoded_secrets로 마킹해서 중복 방지
                    detected_vulns.add("hardcoded_secrets")
                    break  # 첫 번째 하나만 리포트
        
        return findings, score


class LLMAnalyzer:
    """LLM 기반 지능형 분석기"""
    
    def __init__(self):
        self.llm = None
        self._init_llm()
    
    def _init_llm(self):
        """LLM 초기화"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.1,
                    api_key=api_key
                )
                logger.info("LLM analyzer initialized")
            else:
                logger.warning("OpenAI API key not found, LLM analysis disabled")
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
    
    async def analyze(self, code: str, language: str, rule_findings: List[SecurityFinding]) -> Tuple[List[SecurityFinding], str]:
        """
        LLM을 통한 심층 분석
        Returns: (additional_findings, analysis_explanation)
        """
        if not self.llm:
            return [], "LLM analysis not available"
        
        try:
            # Create context from rule-based findings
            context = self._create_analysis_context(rule_findings)
            
            prompt = self._create_analysis_prompt(code, language, context)
            
            response = await self.llm.ainvoke(prompt)
            
            return self._parse_llm_response(response.content)
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return [], f"LLM analysis error: {str(e)}"
    
    def _create_analysis_context(self, rule_findings: List[SecurityFinding]) -> str:
        """규칙 기반 결과로부터 컨텍스트 생성"""
        if not rule_findings:
            return "No initial security concerns detected."
        
        context = "Initial security scan found:\n"
        for finding in rule_findings:
            context += f"- {finding.title}: {finding.description}\n"
        
        return context
    
    def _create_analysis_prompt(self, code: str, language: str, context: str) -> str:
        """LLM 분석을 위한 프롬프트 생성"""
        return f"""
You are a senior cybersecurity expert conducting a thorough code security review.

LANGUAGE: {language}

INITIAL SCAN RESULTS:
{context}

CODE TO ANALYZE:
```{language}
{code}
```

Please provide a comprehensive security analysis focusing on:

1. **Validation of initial findings**: Are the detected issues real vulnerabilities?
2. **Additional vulnerabilities**: What other security issues might exist?
3. **Context-aware assessment**: Consider the broader security implications
4. **Attack scenarios**: How could these vulnerabilities be exploited?
5. **Business impact**: What's the potential damage?

Respond in JSON format:
{{
    "additional_findings": [
        {{
            "rule_id": "string",
            "title": "string", 
            "description": "string",
            "risk_level": "critical|high|medium|low",
            "confidence": 0.0-1.0,
            "cwe_id": "CWE-XXX (if applicable)",
            "attack_scenario": "string",
            "fix_suggestion": "string"
        }}
    ],
    "analysis_summary": "string",
    "overall_assessment": "string",
    "recommended_actions": ["string"]
}}

Focus on accuracy and practical security implications.
"""
    
    def _parse_llm_response(self, response: str) -> Tuple[List[SecurityFinding], str]:
        """LLM 응답 파싱"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                return [], response
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            findings = []
            for item in data.get("additional_findings", []):
                finding = SecurityFinding(
                    rule_id=item.get("rule_id", "llm_finding"),
                    title=item.get("title", "LLM Detection"),
                    description=item.get("description", ""),
                    risk_level=RiskLevel(item.get("risk_level", "medium")),
                    confidence=item.get("confidence", 0.7),
                    cwe_id=item.get("cwe_id"),
                    fix_suggestion=item.get("fix_suggestion")
                )
                findings.append(finding)
            
            analysis = data.get("analysis_summary", response)
            
            return findings, analysis
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return [], response


class HybridSecurityAgent:
    """하이브리드 보안 분석 에이전트"""
    
    def __init__(self):
        self.name = "Hybrid Security Agent"
        self.rule_scanner = RuleBasedScanner()
        self.llm_analyzer = LLMAnalyzer()
        self.graph = self._build_analysis_graph()
    
    def _build_analysis_graph(self) -> StateGraph:
        """분석 워크플로우 그래프 구성"""
        graph = StateGraph(SecurityState)
        
        # 노드 추가
        graph.add_node("detect_language", self._detect_language_node)
        graph.add_node("rule_scan", self._rule_scan_node)
        graph.add_node("decide_depth", self._decide_analysis_depth_node)
        graph.add_node("llm_analysis", self._llm_analysis_node)
        graph.add_node("merge_results", self._merge_results_node)
        graph.add_node("generate_fixes", self._generate_fixes_node)
        
        # 워크플로우 구성
        graph.add_edge("detect_language", "rule_scan")
        graph.add_edge("rule_scan", "decide_depth")
        
        # 조건부 라우팅
        graph.add_conditional_edges(
            "decide_depth",
            self._should_use_llm,
            {
                "llm_analysis": "llm_analysis",
                "skip_llm": "merge_results"
            }
        )
        
        graph.add_edge("llm_analysis", "merge_results")
        graph.add_edge("merge_results", "generate_fixes")
        
        graph.set_entry_point("detect_language")
        graph.add_edge("generate_fixes", END)
        
        return graph.compile()
    
    def _detect_language_node(self, state: SecurityState) -> SecurityState:
        """언어 감지 노드"""
        language = state.input.get("language")
        
        if not language:
            code = state.code
            language = self._detect_language(code)
        
        state.language = language.lower()
        logger.info(f"Detected language: {state.language}")
        return state
    
    def _detect_language(self, code: str) -> str:
        """향상된 언어 감지"""
        # 강력한 키워드 기반 감지
        c_indicators = ["#include", "int main", "char *", "malloc", "printf"]
        if any(indicator in code for indicator in c_indicators):
            return "c"
        
        js_indicators = ["const ", "let ", "var ", "function", "require(", "=>"]
        if any(indicator in code for indicator in js_indicators):
            return "javascript"
        
        python_indicators = ["def ", "import ", "class ", "if __name__"]
        if any(indicator in code for indicator in python_indicators):
            return "python"
        
        # Fallback to pygments
        try:
            lexer = pygments.lexers.guess_lexer(code)
            return lexer.name.lower()
        except:
            return "unknown"
    
    def _rule_scan_node(self, state: SecurityState) -> SecurityState:
        """규칙 기반 스캔 노드"""
        findings, score, needs_llm = self.rule_scanner.scan(state.code, state.language)
        
        state.rule_based_findings = findings
        state.rule_based_score = score
        state.needs_llm_analysis = needs_llm
        
        logger.info(f"Rule scan: {len(findings)} findings, score: {score}%, needs_llm: {needs_llm}")
        return state
    
    def _decide_analysis_depth_node(self, state: SecurityState) -> SecurityState:
        """분석 깊이 결정 노드"""
        # Critical findings or high score triggers deep analysis
        if state.needs_llm_analysis or state.rule_based_score > 50:
            state.analysis_depth = AnalysisDepth.DEEP
        elif state.rule_based_score > 20:
            state.analysis_depth = AnalysisDepth.STANDARD
        else:
            state.analysis_depth = AnalysisDepth.QUICK
        
        logger.info(f"Analysis depth: {state.analysis_depth.value}")
        return state
    
    def _should_use_llm(self, state: SecurityState) -> str:
        """LLM 사용 여부 결정"""
        if state.analysis_depth == AnalysisDepth.DEEP:
            return "llm_analysis"
        return "skip_llm"
    
    async def _llm_analysis_node(self, state: SecurityState) -> SecurityState:
        """LLM 분석 노드"""
        findings, analysis = await self.llm_analyzer.analyze(
            state.code, 
            state.language, 
            state.rule_based_findings
        )
        
        state.llm_findings = findings
        state.llm_analysis = analysis
        
        logger.info(f"LLM analysis: {len(findings)} additional findings")
        return state
    
    def _merge_results_node(self, state: SecurityState) -> SecurityState:
        """결과 병합 노드"""
        # Combine rule-based and LLM findings
        all_findings = state.rule_based_findings + state.llm_findings
        
        # Deduplicate and prioritize
        state.final_findings = self._deduplicate_findings(all_findings)
        
        # Calculate final risk score
        state.risk_score = self._calculate_final_score(state)
        
        logger.info(f"Final analysis: {len(state.final_findings)} findings, score: {state.risk_score}%")
        return state
    
    def _deduplicate_findings(self, findings: List[SecurityFinding]) -> List[SecurityFinding]:
        """중복 제거 및 우선순위 정렬"""
        # Simple deduplication by title similarity
        unique_findings = []
        seen_titles = set()
        
        # Sort by risk level and confidence
        sorted_findings = sorted(
            findings,
            key=lambda f: (f.risk_level.value, f.confidence),
            reverse=True
        )
        
        for finding in sorted_findings:
            if finding.title not in seen_titles:
                unique_findings.append(finding)
                seen_titles.add(finding.title)
        
        return unique_findings
    
    def _calculate_final_score(self, state: SecurityState) -> float:
        """최종 위험도 점수 계산"""
        base_score = state.rule_based_score
        
        # Add LLM findings score
        llm_score = 0
        for finding in state.llm_findings:
            if finding.risk_level == RiskLevel.CRITICAL:
                llm_score += 25
            elif finding.risk_level == RiskLevel.HIGH:
                llm_score += 20
            elif finding.risk_level == RiskLevel.MEDIUM:
                llm_score += 10
        
        return min(base_score + llm_score, 100)
    
    def _generate_fixes_node(self, state: SecurityState) -> SecurityState:
        """수정 제안 생성 노드"""
        if state.risk_score >= state.threshold:
            state.proposed_fix = self._generate_fix_proposals(state)
        
        return state
    
    def _generate_fix_proposals(self, state: SecurityState) -> Dict[str, Any]:
        """수정 제안 생성"""
        fixes = []
        
        for finding in state.final_findings:
            if finding.fix_suggestion:
                fixes.append({
                    "finding": finding.title,
                    "suggestion": finding.fix_suggestion,
                    "priority": finding.risk_level.value
                })
        
        return {
            "strategy": f"Address {len(fixes)} security issues",
            "fixes": fixes,
            "general_recommendations": [
                "Implement input validation",
                "Use parameterized queries",
                "Apply principle of least privilege",
                "Regular security testing"
            ]
        }
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """메인 분석 진입점 (동기)"""
        try:
            # 현재 실행 중인 이벤트 루프가 있는지 확인
            try:
                loop = asyncio.get_running_loop()
                # 이미 실행 중인 루프가 있으면 동기 분석으로 fallback
                return self._sync_analyze(input_data)
            except RuntimeError:
                # 실행 중인 루프가 없으면 새 루프 생성
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    result = loop.run_until_complete(self.async_analyze(input_data))
                    return result
                finally:
                    loop.close()
                
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return self._create_error_response(str(e))
    
    def _sync_analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """동기식 분석 (fallback)"""
        try:
            code = input_data.get("code", "")
            language = input_data.get("language")
            
            # 언어 감지
            if not language:
                language = self._detect_language_sync(code)
            
            # 빠른 규칙 기반 스캔
            findings, risk_score, needs_llm = self.rule_scanner.quick_scan(code, language)
            
            # 임계값 확인
            threshold = input_data.get("metadata", {}).get("threshold", 0.6) * 100
            
            # 취약성 여부 판단 - 높은/치명적 위험도 문제가 있거나 임계값을 넘을 때
            has_high_risk = any(f.risk_level.name in ['HIGH', 'CRITICAL'] for f in findings)
            is_vulnerable = has_high_risk or risk_score >= threshold
            
            # 응답 구성
            return {
                "language": language,
                "risk_score": risk_score,
                "is_vulnerable": is_vulnerable,
                "threshold": threshold,
                "findings": [
                    {
                        "rule": f.rule_id,
                        "detail": f"{f.title}: {f.description}",
                        "line": f.line_number
                    } for f in findings
                ],
                "proposed_fix": None  # 동기 모드에서는 간단한 수정 제안만
            }
            
        except Exception as e:
            logger.error(f"Sync analysis error: {e}")
            return self._create_error_response(str(e))
    
    def _detect_language_sync(self, code: str) -> str:
        """동기식 언어 감지"""
        code_lower = code.lower()
        
        # Python 감지
        python_indicators = ["def ", "import ", "class ", "if __name__", "print(", "sqlite3", "subprocess"]
        if any(indicator in code for indicator in python_indicators):
            return "python"
        
        # JavaScript/Node.js 감지
        js_indicators = ["const ", "let ", "var ", "function(", "require(", "app.post", "express", "jwt", "=>"]
        if any(indicator in code for indicator in js_indicators):
            return "javascript"
        
        # C/C++ 감지
        c_indicators = ["#include", "int main", "char *", "malloc", "printf", "scanf", "gets", "strcpy", "sprintf"]
        if any(indicator in code for indicator in c_indicators):
            return "c"
        
        return "unknown"
    
    async def async_analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """비동기 분석"""
        try:
            # Initialize state
            state = SecurityState(input=input_data)
            
            # Set threshold
            threshold = input_data.get("metadata", {}).get("threshold", 0.6)
            state.threshold = threshold * 100
            
            # Run analysis graph
            result_state = await self.graph.ainvoke(state)
            
            # Build response
            return self._build_response(result_state)
            
        except Exception as e:
            logger.error(f"Async analysis error: {e}")
            return self._create_error_response(str(e))
    
    def _build_response(self, state: SecurityState) -> Dict[str, Any]:
        """응답 구성"""
        findings_data = []
        for finding in state.final_findings:
            findings_data.append({
                "rule": finding.rule_id,
                "detail": f"{finding.title}: {finding.description}",
                "risk_level": finding.risk_level.value,
                "confidence": finding.confidence,
                "line": finding.line_number
            })
        
        return {
            "language": state.language,
            "risk_score": state.risk_score,
            "is_vulnerable": state.risk_score >= state.threshold,
            "threshold": state.threshold,
            "findings": findings_data,
            "proposed_fix": state.proposed_fix,
            "analysis_method": "hybrid",
            "llm_analysis": state.llm_analysis
        }
    
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """에러 응답 생성"""
        return {
            "language": "unknown",
            "risk_score": 0.0,
            "is_vulnerable": False,
            "threshold": 60.0,
            "findings": [{"rule": "error", "detail": error_msg, "line": None}],
            "proposed_fix": None
        }


# 기존 API 호환성을 위한 래퍼
class SecurityAgent(HybridSecurityAgent):
    """기존 SecurityAgent와의 호환성 유지"""
    pass


# 모듈 레벨 API (기존 호환성)
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
    
    parser = argparse.ArgumentParser(description="Hybrid Security Analysis Tool")
    parser.add_argument("--path", help="Path to source file")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    parser.add_argument("--threshold", type=float, default=0.6, help="Vulnerability threshold (0-1)")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON")
    parser.add_argument("--language", help="Override language detection")
    parser.add_argument("--deep", action="store_true", help="Force deep LLM analysis")
    
    args = parser.parse_args()
    
    # Read input
    if args.stdin:
        code = sys.stdin.read()
    elif args.path:
        with open(args.path, 'r') as f:
            code = f.read()
    else:
        print("Error: Provide --path or --stdin")
        sys.exit(1)
    
    # Run analysis
    result = run_security_agent(code, args.language, args.threshold)
    
    # Output
    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))
