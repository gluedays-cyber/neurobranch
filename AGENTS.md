# Agent Behavioral Constraints

## 1. Absolute Ban on SVG Generation
- `assets/`, `docs/assets/`, 또는 프로젝트 내 어떠한 경로에도 `.svg` 파일 생성을 일절 금지한다.
- 인라인 `<svg>` 코드 삽입을 금지하며, 시각화는 Mermaid(` ```mermaid `) 또는 마크다운 텍스트 블록만을 사용한다.

## 2. Autonomous Execution
- 사용자의 중간 승인 대기 없이 단일 턴 내에서 전수 검증까지 자율 완결 집행한다.
