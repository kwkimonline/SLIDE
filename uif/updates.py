import csv


def update_perfs(perfs, learning_stats) :

    acc, bacc, con, con10, con50 = perfs

    learning_stats["acc"].append(acc)
    learning_stats["bacc"].append(bacc)
    learning_stats["con"].append(con)
    learning_stats["con10"].append(con10)
    learning_stats["con50"].append(con50)

    return learning_stats

    
def write_perfs(result_path, file_name, mode, lr, epochs, opt, lmda, gamma, tau, learning_stats) :

    file = open(result_path + file_name, "a")
    writer = csv.writer(file)
    writer.writerow([mode, lr, epochs, opt, lmda, gamma, tau, 
                    learning_stats["acc"][-1], learning_stats["bacc"][-1], 
                    learning_stats["con"][-1], learning_stats["con10"][-1], learning_stats["con50"][-1]
                    ])
    file.close()
