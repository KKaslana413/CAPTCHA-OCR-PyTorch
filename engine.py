import torch
import torch.nn as nn
from config import CHAR_LIST, CONFIDENCE_THRESHOLD, ENTROPY_WEIGHT

Loss_Function = nn.CTCLoss(blank=0, zero_infinity=True)


def Greedy_Decode(Predict_Seq, confidence_threshold=CONFIDENCE_THRESHOLD):
    Predict_List = []
    Predict_Seq = Predict_Seq.permute(1, 0, 2)
    probs = torch.softmax(Predict_Seq, dim=2)
    max_vals, max_idx = torch.max(probs, dim=2)

    for vals, seq in zip(max_vals, max_idx):
        Char_List = []
        previous = -1
        for prob, idx in zip(vals.tolist(), seq.tolist()):
            if idx == previous or idx == 0:
                previous = idx
                continue
            if prob < confidence_threshold:
                previous = idx
                continue
            Char_List.append(CHAR_LIST[idx - 1])
            previous = idx
        Predict_List.append("".join(Char_List))
    return Predict_List


def entropy_regularization(logits):
    probs = torch.exp(logits)
    entropy = - (probs * torch.log(probs + 1e-8)).sum(dim=2).mean()
    return entropy


def levenshtein(a, b):
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n]


def Train_Fn(Model, Data_Loader, Optimizer, Device):
    Model.train()
    Total_Loss = 0.0

    for batch_idx, (Images, Labels) in enumerate(Data_Loader):
        Images = Images.to(Device)
        Labels = [label.to(Device) for label in Labels]

        main_pred, aux_pred = Model(Images)  # two heads
        Targets = torch.cat(Labels)

        Target_Lengths = torch.tensor([len(label) for label in Labels], dtype=torch.long).to(Device)
        Batch_Size = Images.size(0)
        Input_Lengths = torch.full(size=(Batch_Size,), fill_value=main_pred.size(0), dtype=torch.long).to(Device)

        loss_main = Loss_Function(main_pred, Targets, Input_Lengths, Target_Lengths)
        loss_aux = Loss_Function(aux_pred, Targets, Input_Lengths, Target_Lengths)
        entropy_pen = entropy_regularization(main_pred)

        # deep supervision: 主 loss + 辅助 loss（权重 0.3 可调） - entropy 正则
        loss = loss_main + 0.3 * loss_aux - ENTROPY_WEIGHT * entropy_pen

        Optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(Model.parameters(), max_norm=5.0)
        Optimizer.step()

        Total_Loss += loss.item()
        if batch_idx % 50 == 0:
            print(f"[Train] Batch {batch_idx}/{len(Data_Loader)} Loss: {loss.item():.4f}")

    return Total_Loss / len(Data_Loader)


def Eval_Fn(Model, Data_Loader, Device):
    Model.eval()
    Total = 0
    Correct = 0
    Char_Correct, Char_Total = 0, 0
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for batch_idx, (Images, Labels) in enumerate(Data_Loader):
            Images = Images.to(Device)
            Labels = [label.to(Device) for label in Labels]

            main_pred, _ = Model(Images)
            Predict_Strings = Greedy_Decode(main_pred)

            Target_Strings = []
            for label in Labels:
                Target_Strings.append("".join([CHAR_LIST[c.item() - 1] for c in label]))

            if batch_idx == 0:
                for i in range(min(20, len(Predict_Strings))):
                    print(f"[Eval Debug] Sample {i+1}: Pred={Predict_Strings[i]} | True={Target_Strings[i]}")

            for Pred_Str, True_Str in zip(Predict_Strings, Target_Strings):
                for p, t in zip(Pred_Str, True_Str):
                    if p == t:
                        Char_Correct += 1
                    Char_Total += 1
                if Pred_Str == True_Str:
                    Correct += 1
                Total += 1
                all_preds.append(Pred_Str)
                all_targets.append(True_Str)

    accuracy = Correct / Total if Total > 0 else 0.0
    total_edits = 0
    total_chars = 0
    for p, t in zip(all_preds, all_targets):
        total_edits += levenshtein(p, t)
        total_chars += len(t)
    cer = total_edits / max(1, total_chars)

    print(f"Eval: Exact Match Acc={accuracy:.4f}, CER={cer:.4f}")
    return accuracy
