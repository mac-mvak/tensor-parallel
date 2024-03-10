## Реализация.

В качестве базовой модели я взял модель из предыдущего домашнего задания, свёрточную сеть с линейными слоями, батчнормой, макспулингами и линейными слоями.


Теперь я расскажу про архитектуру разбиения, и как оно было реализовано. 

Разделял я модель по выходным нейронам. Нетрудно заметить, что тогда нам можно не изменять каким-то образом линейные или иные слои, а достаточно написать 2 
новых слоя, первый из которых будет собирать в единый тензор после того, как мы посчитали части на форварде, а на бэкварде оставлять только часть градиента, 
отвечающую куску который мы получили на вход в форвард. Этот слой это `Gather1` в `gathers.py`. Второй слой это слой, отвечающий за правильный сбор градиентов ото 
всех частей. На форварде он просто пропускает через себя данный массив, а на бэкварде он делает `all_reduce` по всем градиентов из разных процессов (это правило 
полной производной).  Этот слой это `GatherGrad1` в `gathers.py`. Здесь все слои делают объединение по первой координате, и я решил это захардкодить, так как мы 
делим по output, который отвечает за первую координату. Тогда я разделил всю модель на 4 куска, в которых в начале подал `GatherGrad1`, а затем `Gather1`. А затем, 
просто использовал слои, просто с уменьшим в `size` раз выходным каналом. У нас нет слоёв, которые бы взаимодействовали со всеми каналами, поэтому нам достаточно 
вот этих двух классов, которые будут обеспечивать протекание градиентов. Модель для многопоточного обучения это `model_shards.py`.

Теперь опишем как мы обрабатываем большой батч. Тут всё делает куда проще, чем в прошлом пункте.

Первый процесс читает батч, отдаёт `batch_size`, а также сам батч и инпуты на все потоки. Далее мы просто делаем обычный форвард и обычный бэквард, наши слои 
сделают протекание градиентов автоматическим и нам не надо более модифицировать само обучение и тестирование модели.



Все циклы обучения лежат в файле `loops.py`.

## Результаты.

Обычная модель запускается кодом `python model.py`.

Чтобы запустить распределённый код, достаточно запустить команду:

```
python3 distr_model.py -b BATCH_SIZE -s SPLIT_SIZE
```

`BATCH_SIZE` -- размер большого батча, `SPLIT_SIZE` -- на сколько потоков мы делим модель.


Результаты ранов: [ссылка](https://wandb.ai/svak/tensor_parallel)


Видно что модели учатся по сути дела одинаково, то есть разделение никак не ухудшило результаты.


