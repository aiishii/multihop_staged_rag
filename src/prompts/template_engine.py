from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path

# テンプレートエンジンの初期化
template_path = Path(__file__).parent / "templates"
env = Environment(
    loader=FileSystemLoader(template_path),
    autoescape=select_autoescape()
)

def render_template(template_name, **kwargs):
    """指定されたテンプレートをレンダリング"""
    template = env.get_template(f"{template_name}.j2")
    return template.render(**kwargs)