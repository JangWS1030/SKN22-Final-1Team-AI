"""
MirrAI SD Inpainting — postprocess (분리 완료)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
이 모듈은 세 도메인으로 분리되었습니다:

  mask_builders.py   — _build_*_mask() 계열 함수 전체
  cloth_preserve.py  — 옷 보존/복원 로직
  refinement.py      — SD/CV2/LaMa 정제 + composite + unload

backward compat: bind_postprocess_methods_to_pipeline은 그대로 유지되며
세 모듈의 bind 함수를 순서대로 호출합니다.
"""

from .mask_builders import bind_mask_builder_methods_to_pipeline
from .cloth_preserve import bind_cloth_preserve_methods_to_pipeline
from .refinement import bind_refinement_methods_to_pipeline


def bind_postprocess_methods_to_pipeline(cls) -> None:
    """postprocess 메서드 전체를 MirrAISDPipeline에 바인딩 (분리된 3개 모듈 위임)."""
    bind_mask_builder_methods_to_pipeline(cls)
    bind_cloth_preserve_methods_to_pipeline(cls)
    bind_refinement_methods_to_pipeline(cls)
