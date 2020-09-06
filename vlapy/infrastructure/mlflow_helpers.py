# MIT License
#
# Copyright (c) 2020 Archis Joglekar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import shutil
import json

from mlflow.tracking import MlflowClient


def get_this_metric_of_this_run(metric_name, run_object):
    client = MlflowClient()
    run_id = run_object.info.run_id

    run = client.get_run(run_id)
    return run.data.metrics[metric_name]


def download_run_artifacts_for_resume(path, run_id):
    client = MlflowClient()
    run = client.get_run(run_id=run_id)
    artifact_uri = run.info.artifact_uri

    if "s3" in artifact_uri:
        os.system("aws s3 sync " + artifact_uri + " " + path)
    else:
        copytree_alt(artifact_uri[7:], path)

    with open(os.path.join(path, "all_parameters.txt"), "r") as fp:
        old_params = json.load(fp)

    return old_params


def copytree_alt(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
