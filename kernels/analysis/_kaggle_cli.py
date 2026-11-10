"""Locate the Kaggle CLI on a cloud checkout or an existing Mac installation."""
import os
import shutil
from pathlib import Path


def kaggle_cli():
    override = os.environ.get('KAGGLE_BIN')
    if override:
        executable = shutil.which(str(Path(override).expanduser()))
        if executable:
            return executable
        raise SystemExit('KAGGLE_BIN does not name an executable Kaggle CLI. Fix it or unset it.')

    executable = shutil.which('kaggle')
    if executable:
        return executable

    home = Path.home()
    candidates = [home / '.local/bin/kaggle',
                  *sorted((home / 'Library/Python').glob('*/bin/kaggle'), reverse=True)]
    for candidate in candidates:
        executable = shutil.which(str(candidate))
        if executable:
            return executable

    raise SystemExit('Kaggle CLI not found. Install kaggle and add it to PATH, or set KAGGLE_BIN to its executable.')
