import os
import datetime
import shutil

DIRS = ['configs', 'core', 'tools', 'scripts']

def backup_codes(path):
    root_path=os.path.abspath(os.path.join(os.getcwd()))
    names = DIRS

    path = os.path.join(path, "code_{}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    os.makedirs(path, exist_ok=True)

    for name in names:
        if os.path.exists(os.path.join(root_path, name)):
            shutil.copytree(os.path.join(root_path, name), os.path.join(path, name))

    pyfiles = filter(lambda x: x.endswith(".py") or x.endswith(".sh"), os.listdir(root_path))
    for pyfile in pyfiles:
        shutil.copy(os.path.join(root_path, pyfile), os.path.join(path, pyfile))