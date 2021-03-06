# HPC_VectorSum_Korshikov
 
Суммирование элементов вектора

Задание на лабораторную и аппаратная база
Задача: реализовать алгоритм сложения элементов вектора.
Входные данные: 1 вектор размером от 1_000 до 1_000_000.
Выходные данные: сумма элементов вектора + время вычисления
Реализация должна содержать 2 функции сложения элементов вектора: на CPU и на GPU с применением CUDA.

Вычисления проводились на ноутбуке со следующей конфигурацией:
Центральный процессор: Intel I5-11300H @ 3,10 GHz.
Оперативная память: Samsung DDR4, 8 GB, 3200 MHz, SingleChannel.
Графический процессор: RTX 3060, 6GB VRAM GDDR6.
Алгоритм
Необходима определиться с механизмом реализации данного алгоритма.
Для работы с GPU за основу взята функция sum_reduction. 
Для реализации данного алгоритма применяется разделяемая память, так как её показатели являются компромисом по сравнению с логальной и глобальной памятью. Работа функции sum_reduction основана на редукции.
То есть, для вычисления элементов массива мы параллельно вычисляем суммы отдельных элементов массива.



Но, есть способ повысить эффективность вычислений. Нам необходимо разделить блок пополам и складывать элементы из половин блока.



Равспараллелив этот процесс, можно добиться ещё более существенного ускорения по сравнению с CPU. Распараллеливанию подлежат целые целые warp-ы, а когда останется меньше 32 потоков, то подсчёт будет осуществлять функция устройства. Это позволит добиться лучшей оптимизации, так как это избавит нас от необходимости вызывать функцию синхронизации syncthreads() в данном случае. На CPU сложение элементов вектора, который, выполняется с помощью функции,  sumVectorCPU(), которая рассматривает вектор как одномерный массив. 
Результаты работы программы
С результатами лабораторной можно ознакомиться в файле vector_results.xlsx.



Опираясь на результаты испытаний можно сделать вывод, что использование GPU для вычисления сравнительно небольшого вектора не имеет практического смысла. GPU и CPU в моей сборке смогли сравниться по производительности только при размерности вектора 2^13. Отчасти, это связано с тем, что необходимо затратить время на то, чтобы распараллелить алгоритм и синхронизировать потоки. По мере увеличения размерности вектора, преимущество параллельных вычислений на GPU над последовательными вычислениями на CPU становится всё нагляднее и нагляднее.
