arr = [0,1,3,5,7,8,9,10]

def erfen(arr:list, x: int):
    left, right = 0, len(arr)-1
    return erfen_sub(arr, left, right, x)


def erfen_sub(arr:list, left:int, right:int, x:int):
    if left < right:
        ret1 = erfen_sub(arr, left, (left+right)//2, x)
        ret2 = erfen_sub(arr, (left+right)//2 + 1, right, x)
        return ret1 if ret1 else ret2
    elif arr[left] == x:
        return left 
    else:
        return None
    


print(erfen(arr, 1))
