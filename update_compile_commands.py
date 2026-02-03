import json
import os
import sys

# Dynamically determine project root relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
COMPILE_COMMANDS_PATH = os.path.join(PROJECT_ROOT, "compile_commands.json")

# System include paths - these must remain absolute as they are system specific
SYSTEM_INCLUDES = [
    "-I/usr/include/c++/11",
    "-I/usr/include/x86_64-linux-gnu/c++/11",
    "-I/usr/include/c++/11/backward",
    "-I/usr/lib/gcc/x86_64-linux-gnu/11/include",
    "-I/usr/local/include",
    "-I/usr/include/x86_64-linux-gnu",
    "-I/usr/include"
]

def get_all_source_files(root_dir):
    source_files = []
    for root, dirs, files in os.walk(root_dir):
        # Exclude hidden directories and build artifacts not relevant to source
        if ".git" in dirs: dirs.remove(".git")
        if ".vscode" in dirs: dirs.remove(".vscode")
        if "bin" in dirs: dirs.remove("bin")
        if "runs" in dirs: dirs.remove("runs")
        
        for file in files:
            if file.endswith((".cc", ".cpp")):
                source_files.append(os.path.abspath(os.path.join(root, file)))
    return source_files

def to_relative(path, root):
    """Converts an absolute path to relative if it's inside root."""
    try:
        # Check if path is actually inside root to avoid ../../../ mess if not intended
        # os.path.relpath can return ../.. if outside. 
        # We only want to convert if it starts with root.
        abs_path = os.path.abspath(path)
        abs_root = os.path.abspath(root)
        
        if abs_path.startswith(abs_root):
            return os.path.relpath(abs_path, abs_root)
        return path
    except ValueError:
        return path

def main():
    if not os.path.exists(COMPILE_COMMANDS_PATH):
        print(f"Error: {COMPILE_COMMANDS_PATH} not found.")
        return

    try:
        with open(COMPILE_COMMANDS_PATH, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Error: compile_commands.json is invalid JSON.")
        return

    # Map existing entries by absolute file path for checking existence
    # We might need to handle the case where 'file' is already relative in the JSON (unlikely from compiledb but possible)
    existing_files_set = set()
    for entry in data:
        fpath = entry['file']
        if not os.path.isabs(fpath):
            fpath = os.path.abspath(os.path.join(PROJECT_ROOT, fpath))
        existing_files_set.add(fpath)
    
    # Use the first entry as a template for flags
    if not data:
        print("Error: compile_commands.json is empty.")
        return

    # Extract base flags and make them relative if possible
    template_entry = data[0]
    base_flags = []
    args = template_entry['arguments']
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '-c':
            i += 2 
        elif arg == '-o':
            i += 2 
        else:
            # Check if arg is an include path to relativize
            if arg.startswith("-I"):
                path = arg[2:]
                rel_path = to_relative(path, PROJECT_ROOT)
                base_flags.append(f"-I{rel_path}")
            else:
                base_flags.append(arg)
            i += 1

    # Ensure system includes are in the base flags 
    for inc in SYSTEM_INCLUDES:
        if inc not in base_flags:
            base_flags.append(inc)

    # 1. Update existing entries
    print("Updating existing entries (relativizing paths)...")
    for entry in data:
        current_args = entry['arguments']
        new_args = []
        i = 0
        while i < len(current_args):
            arg = current_args[i]
            if arg == '-c':
                new_args.append(arg)
                # Relativize source file path in args
                src_file = current_args[i+1]
                new_args.append(to_relative(src_file, PROJECT_ROOT))
                i += 2
            elif arg == '-o':
                new_args.append(arg)
                # Relativize output file path in args
                out_file = current_args[i+1]
                new_args.append(to_relative(out_file, PROJECT_ROOT))
                i += 2
            elif arg.startswith("-I"):
                path = arg[2:]
                rel_path = to_relative(path, PROJECT_ROOT)
                new_args.append(f"-I{rel_path}")
                i += 1
            else:
                new_args.append(arg)
                i += 1
        
        # Append system includes if missing
        for inc in SYSTEM_INCLUDES:
            if inc not in new_args:
                new_args.append(inc)
                
        entry['arguments'] = new_args
        entry['file'] = to_relative(entry['file'], PROJECT_ROOT)
        # Note: 'directory' is left absolute intentionally to avoid breaking clangd which expects an absolute working directory

    # 2. Add missing files
    all_sources = get_all_source_files(PROJECT_ROOT)
    added_count = 0
    
    print("Checking for missing files...")
    for source_file in all_sources:
        if source_file not in existing_files_set:
            print(f"Adding: {to_relative(source_file, PROJECT_ROOT)}")
            
            # Construct command line arguments
            new_args = list(base_flags) 
            new_args.append("-c")
            new_args.append(to_relative(source_file, PROJECT_ROOT))
            
            # Fake output path
            rel_name = os.path.basename(source_file) + ".o"
            new_args.append("-o")
            new_args.append(to_relative(os.path.join(PROJECT_ROOT, "bin/x86_64/generated", rel_name), PROJECT_ROOT))
            
            new_entry = {
                "directory": PROJECT_ROOT, 
                "arguments": new_args,
                "file": to_relative(source_file, PROJECT_ROOT)
            }
            data.append(new_entry)
            added_count += 1

    # Save back to file
    with open(COMPILE_COMMANDS_PATH, 'w') as f:
        json.dump(data, f, indent=1)
    
    print(f"Done. Updated {len(data) - added_count} existing entries and added {added_count} new entries.")

if __name__ == "__main__":
    main()
