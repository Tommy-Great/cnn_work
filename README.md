
## 项目运行方法

python运行函数:
```python
import run_cnn
run_cnn.run_model(run_type = 'pretrain', epochs = 10, 
              bs_list = (32, 64, 128), 
              lr_list = (5e-4, 1e-4, 5e-5))
``` 

