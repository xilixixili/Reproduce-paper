def compute_accuracy(model , batch_s , num_classes , data_loader) :  # probas：sofamax输出向量
    correct_pred , num_examples = 0 , 0
    for features , targets in data_loader :
        # print(type(targets))
        features = features.to(device)
        targets = targets.to(device)
        logits = model(features)
        probas = F.softmax(logits , dim=1)
        _ , predicted = torch.max(probas , 1)  # 返回最大值及索引
        num_examples += targets.size(0)
        correct_pred += (predicted == targets).sum()
    return correct_pred.float() / num_examples * 100


def compute_mis(model , data_loader) :
    #mis = np.array([[0 , 0 , 0 , 0 , 0] , [0 , 0 , 0 , 0 , 0] , [0 , 0 , 0 , 0 , 0] , [0 , 0 , 0 , 0 , 0] ,
    #                [0 , 0 , 0 , 0 , 0]])  # [target][predicted]
    # mis={'one2one':0,'one2two':0,'one2three'}
    mis = np.zeros([10,10])
    correct_pred , num_examples = 0 , 0
    for features , targets in data_loader :
        print(targets)
        features = features.to(device)
        targets = targets.to(device)
        logits = model(features)
        probas = F.softmax(logits , dim=1)
        _ , predicted = torch.max(probas , 1)  # 返回最大值及索引

        predicted = predicted.int()
        targets = targets.int()
        # print('=====',targets[0].item())
        lens = len(targets)
        for i in range(lens) :
            # print('i:',i,'targets[i].item()',targets[i].item(),'predicted[i].item()',predicted[i].item())
            mis[targets[i].item()][predicted[i].item()] += 1
        num_examples += targets.size(0)
        correct_pred += (predicted == targets).sum()

    print('Total average acc: %.2f%%' % (correct_pred.float() / num_examples * 100))
    row , col = mis.shape
    sum_row = np.sum(mis , axis=1)
    print(sum_row)
    print('===============================')
    print(mis)
    for ro in range(row) :
        for co in range(col) :
            print(ro , 'To' , co , ':' , '%5.2f%%' % (mis[ro][co] / sum_row[ro] * 100) , end='  ')
        print()