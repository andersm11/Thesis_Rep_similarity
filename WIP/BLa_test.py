num_test=100
Matrix_field_val = np.zeros(num_test)
for i in range(num_test):
    print(i)
    calculator = CKACalculator(model1=mods[0][0], model2=mods[1][0], dataloader=train_loader,
                                    layers_to_hook=(nn.Conv2d, nn.Linear, nn.AdaptiveAvgPool2d, GraphConvolution, nn.BatchNorm1d))
    cka_output = calculator.calculate_cka_matrix()
    Matrix_field_val[i]=cka_output[0][4]