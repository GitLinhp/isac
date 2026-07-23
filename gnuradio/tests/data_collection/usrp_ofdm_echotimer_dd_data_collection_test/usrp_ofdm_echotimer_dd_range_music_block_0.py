import sys
from pathlib import Path
for _p in [Path.cwd(), *Path.cwd().parents]:
    _src = _p / "src"
    if (_src / "isac_imp").is_dir():
        sys.path.insert(0, str(_src))
        break
from isac_imp.range_music_block import RangeMusicBlock

blk = RangeMusicBlock
