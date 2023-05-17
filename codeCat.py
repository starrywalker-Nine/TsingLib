import os
import glob

def concatenate_files(folder_path, output_file_name):
    # Ensure the output file is not in the directories being scanned
    assert not os.path.exists(os.path.join(folder_path, output_file_name)), \
        f"Output file {output_file_name} should not be in the target directories!"

    # Open the output file
    with open(output_file_name, 'w') as outfile:
        # Walk through each directory in the folder
        for dirpath, dirnames, filenames in os.walk(folder_path):
            # Find any .py files in this directory
            for file_name in glob.glob(os.path.join(dirpath, "*.py")):
                # Do not include the output file in the concatenation
                if file_name == output_file_name:
                    continue
                # Open each .py file and append its contents to the output file
                with open(file_name, 'r') as infile:
                    outfile.write(f'# File: {file_name}\n')
                    outfile.write('# Code:\n')
                    outfile.write('# ' + '-' * 100 + '\n')
                    outfile.write('# ' + '-' * 100 + '\n')
                    outfile.write('# ' + '-' * 100 + '\n')
                    outfile.write(infile.read())
                    # Add a newline after each file to ensure scripts don't run together
                    outfile.write('\n')
        outfile.write('请使用中文告诉我这段代码的核心内容并帮助我进行理解，我会问你一些函数内部的问题，你需要帮我进行代码修改并回答我的问题，谢谢！')

# Use the function 使用实例如下，替换<YOUR FOLDER PATH HERE>为你的文件夹绝对路径，替换output.py为你的输出文件名，再把输出文件进行上传即可
concatenate_files('<YOUR FOLDER PATH HERE>', 'output.py')
# concatenate_files('/home/starrywalker/text-test/source', 'output.py')