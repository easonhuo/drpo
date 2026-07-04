on open droppedItems
    set applicationPath to POSIX path of (path to me)
    set runnerPath to applicationPath & "Contents/Resources/run_update.sh"

    tell application "Terminal"
        activate
        repeat with itemReference in droppedItems
            set packagePath to POSIX path of itemReference
            set commandText to "/bin/bash " & quoted form of runnerPath & " " & quoted form of packagePath
            do script commandText
        end repeat
    end tell
end open

on run
    tell application "Terminal"
        activate
        do script "printf '%s\\n' '请双击一个 .drpoupdate 更新文件。更新过程会在终端中显示。'"
    end tell
end run
