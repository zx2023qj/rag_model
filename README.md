# 食用方法

如果已经安装了neo4j和dify，只需修改pdf所在的路径，然后

```
python pdf_to_neo4j.py
python search_moudle_for_dify.py
```

然后再在dify搭建工作流即可。

具体流程可见blog：https://zx2023qj.github.io/2025/05/25/how-to-build-rag-model-by-dify-and-neo4j-md/

由于pyproject.toml需要手动配置，所以懒得配，看requirements.txt罢

补充关于test_and_evaluate.py的说明，由于一些原因，这些部分的功能是不完全的，因此只作参考。

测评集在testdata文件夹中。