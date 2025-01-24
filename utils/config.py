class Config:
    """从py文件中加载变量, 并转换成字典形式"""
    @staticmethod
    def fromfile(file_dir: str):
        config_dict = {}
        with open(file_dir) as f:
            exec(f.read(), config_dict)
        return {k: v for k, v in config_dict.items() if not k.startswith('_')}