"""Versioned declarative skills. Archives are read as data, never extracted/executed."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile
from app import config

MAX_BYTES = 2 * 1024 * 1024
MAX_TEXT_BYTES = 240000
DEFAULT = Path(__file__).resolve().parents[1] / 'skill_runtime' / 'default_external_skill.json'

def parse_upload(filename, data):
    if not data or len(data) > MAX_BYTES:
        raise ValueError('Skill 文件需为 1 字节至 2 MB')
    documents = {}
    if filename.lower().endswith('.zip'):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                entries = archive.infolist()
                if len(entries) > 200 or sum(i.file_size for i in entries) > 4 * MAX_BYTES:
                    raise ValueError('Skill 压缩包展开后过大')
                for item in entries:
                    path = PurePosixPath(item.filename)
                    if path.is_absolute() or '..' in path.parts or '\\' in item.filename or (item.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError('Skill 包含不安全路径')
                    if item.is_dir() or '__MACOSX' in path.parts or path.name.startswith('.'):
                        continue
                    if path.suffix.lower() == '.md':
                        if item.file_size > MAX_TEXT_BYTES:
                            raise ValueError('Skill 文档过大')
                        documents[str(path)] = archive.read(item).decode('utf-8-sig')
        except (zipfile.BadZipFile, UnicodeError, RuntimeError) as exc:
            raise ValueError('请上传有效的 UTF-8 Skill ZIP') from exc
        mains = [p for p in documents if PurePosixPath(p).name.lower() == 'skill.md']
        if len(mains) != 1:
            raise ValueError('ZIP 中需有且仅有一个 SKILL.md')
        main = mains[0]
    elif filename.lower().endswith('.md'):
        try:
            documents = {'SKILL.md': data.decode('utf-8-sig')}
        except UnicodeError as exc:
            raise ValueError('Skill 必须为 UTF-8 文本') from exc
        main = 'SKILL.md'
    else:
        raise ValueError('仅支持 .md 或 .zip Skill')
    text = '\n\n'.join(f'## {p}\n{documents[p]}' for p in [main, *sorted(p for p in documents if p != main)])
    if not documents[main].strip() or len(text.encode()) > MAX_TEXT_BYTES:
        raise ValueError('Skill 为空或 Markdown 总量超过 240 KB')
    return {'id': hashlib.sha256(text.encode()).hexdigest(), 'name': Path(filename).name[:120], 'text': text,
            'documents': list(documents), 'format': 'declarative-markdown-v1'}

def save(filename, data):
    value = parse_upload(filename, data)
    folder = config.data_dir() / 'external_skills'
    folder.mkdir(exist_ok=True)
    path = folder / (value['id'] + '.json')
    # Content-addressed, immutable: concurrent writes have identical content.
    if not path.exists():
        path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    return info(value)

def load(skill_id=None):
    if not skill_id or skill_id == 'default':
        return json.loads(DEFAULT.read_text())
    if len(skill_id) != 64 or any(c not in '0123456789abcdef' for c in skill_id):
        raise ValueError('Skill 编号无效')
    path = config.data_dir() / 'external_skills' / (skill_id + '.json')
    if not path.is_file():
        raise ValueError('Skill 不存在，请重新上传')
    return json.loads(path.read_text())

def info(value):
    return {k: value[k] for k in ('id', 'name', 'documents', 'format')}


def execution_text(value):
    """Use a reviewed visual-only profile for the bundled skill; preserve custom text."""
    default = load()
    if value['id'] == default['id']:
        return DEFAULT.with_name('default_external_instructions.md').read_text(encoding='utf-8')
    return value['text']
