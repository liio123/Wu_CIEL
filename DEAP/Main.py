from DEAP.DataLoad import *
from DEAP.Model import *
from Index_calculation import *


# ---------------------------------------------
# Loss Functions: Classification + Physics + VMC
# ---------------------------------------------


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# batch_size = 256
num_epochs = 200
learning_rate = 0.005
lambda_phys = 0.5
lambda_vmc = 0.5

average_acc = 0
for train_idx, test_idx in kfold.split(MyDataset):
    train_data = Subset(MyDataset, train_idx)
    test_data = Subset(MyDataset, test_idx)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, drop_last=True)
    min_acc = 0.3
    model = PINN_VMC(hidden_dim=32, output_dim=2, xdim=[batch_size, 32, 5], kadj=2, num_out=16, dropout=0.5).to(device)
    # loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    def cross_entropy_loss(logits, labels):
        return F.cross_entropy(logits, labels)
    # loss_phys = λ * ||∇output/∇x||²;「梯度平滑」形式的正则项
    def physics_loss_fn(model, x, fadj):
        # 强制设置整个模型为训练模式
        model.train()
        # 手动设置所有 BatchNorm1d 层为训练模式
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.train()

        # 确保 x 具有梯度追踪
        x = x.clone().detach().requires_grad_(True)

        # 前向传播得到输出
        output = model(x, fadj)

        # 将输出转换为标量，这里使用 sum() 保证所有元素参与计算图
        scalar_output = output.sum()

        # # 调试输出以确认梯度需求
        # print("x.requires_grad:", x.requires_grad)  # 应为 True
        # print("output.requires_grad:",
        #       output.requires_grad)  # 应为 True\n    print(\"scalar_output.requires_grad:\", scalar_output.requires_grad)  # 应为 True

        # 计算梯度
        grad_output = torch.autograd.grad(
            outputs=scalar_output,
            inputs=x,
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0]

        # print("grad_output shape:", grad_output.shape)

        loss = torch.mean(grad_output ** 2)
        return loss

    def vmc_loss_fn(model, x, pcc):
        noise = torch.randn_like(x) * 0.01
        sampled_x = x + noise
        output = model(sampled_x, pcc)
        return torch.mean(output ** 2)
    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []

    G = testclass()
    train_len = G.len(train_idx.shape[0], batch_size)
    test_len = G.len(test_idx.shape[0], batch_size)

    for epoch in range(num_epochs):
    # -------------------------------------------------
        total_train_acc = 0
        total_train_loss = 0

        for data, pcc, labels in train_dataloader:
            data = data.to(device)
            pcc = pcc.to(device)
            labels = labels.to(device)
            # print(labels.shape)

            output = model(data, pcc)
            # print("output:", output.shape)
            # print(labels.shape)

            # train_loss= cross_entropy_loss(output, labels.long())

            class_loss = cross_entropy_loss(output, labels.long())
            physics_loss = physics_loss_fn(model, data, pcc)
            vmc_loss = vmc_loss_fn(model, data, pcc)
            train_loss = class_loss + lambda_phys * physics_loss + lambda_vmc * vmc_loss

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            train_acc = (output.argmax(dim=1) == labels).sum()

            train_loss_list.append(train_loss)
            total_train_loss = total_train_loss + train_loss.item()

            train_acc_list.append(train_acc)
            total_train_acc += train_acc

        train_loss_list.append(total_train_loss / (len(train_dataloader)))
        train_acc_list.append(total_train_acc / train_len)

    # -------------------------------------------------
        total_test_acc = 0
        total_test_loss = 0


        for data, pcc, labels in test_dataloader:
            data = data.to(device)
            pcc = pcc.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                output = model(data, pcc)
                # test_loss = loss_func(output, labels.long())

                class_loss = cross_entropy_loss(output, labels.long())
            physics_loss = physics_loss_fn(model, data, pcc)
            vmc_loss = vmc_loss_fn(model, data, pcc)
            test_loss = class_loss + lambda_phys * physics_loss + lambda_vmc * vmc_loss

            test_acc = (output.argmax(dim=1) == labels).sum()

            test_loss_list.append(class_loss)
            total_test_loss = total_test_loss + class_loss.item()

            test_acc_list.append(test_acc)
            total_test_acc += test_acc

        test_loss_list.append(total_test_loss / (len(test_dataloader)))
        test_acc_list.append(total_test_acc / test_len)

        if (total_test_acc / test_len) > min_acc:
            min_acc = total_test_acc / test_len
            # res_TP_TN_FP_FN = TP_TN_FP_FN
            torch.save(model.state_dict(), 'G:\吴老师的论文/2025NIPS\保存的模型\DEAP/arousal/1.pth')

        # print result
        print("Epoch: {}/{} ".format(epoch + 1, num_epochs),
              "Training Loss: {:.4f} ".format(total_train_loss / len(train_dataloader)),
              "Training Accuracy: {:.4f} ".format(total_train_acc / train_len),
              "Test Loss: {:.4f} ".format(total_test_loss / len(test_dataloader)),
              "Test Accuracy: {:.4f}".format(total_test_acc / test_len)
              )
    print(min_acc)
    average_acc += min_acc
average_acc = average_acc / 5
print("平均准确率为：", average_acc)