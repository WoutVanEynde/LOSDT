import os
import sys
import streamlit.web.bootstrap

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    flag_options = {
        "server.port": 8501,
        "global.developmentMode": False,
        "server.fileWatcherType": "none",  # Disable file watching to avoid inotify errors
        "server.runOnSave": False,  # Disable auto-reload
    }

    streamlit.web.bootstrap.load_config_options(flag_options=flag_options)
    flag_options["_is_running_with_streamlit"] = True
    streamlit.web.bootstrap.run(
        "LOSDT.py",
        False,
        ['run'],
        flag_options,
    )
