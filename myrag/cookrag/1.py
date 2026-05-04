import os
import shutil

def move_md_files(source_dir, target_dir):
    # 如果目标目录不存在，则创建它
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目录: {target_dir}")

    count = 0
    # os.walk 会递归遍历文件夹
    print(f'??? {source_dir}')
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            print(file)
            if file.endswith('.md'):
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, file)
                
                # 如果目标文件夹已存在同名文件，防止覆盖（可选：重命名）
                if os.path.exists(target_path):
                    name, ext = os.path.splitext(file)
                    target_path = os.path.join(target_dir, f"{name}_副本{ext}")

                try:
                    shutil.move(source_path, target_path)
                    print(f"已移动: {file}")
                    count += 1
                except Exception as e:
                    print(f"移动 {file} 出错: {e}")

    print(f"\n任务完成！共移动了 {count} 个 .md 文件到 {target_dir}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 配置路径
    source_dir = os.path.join(BASE_DIR, 'HowToCook-master','tips')  # 搜索当前目录
    destination_folder =os.path.join(BASE_DIR, 'data')  # 移动到的目标目录
    print(source_dir)
    
    move_md_files(source_dir, destination_folder)
