# 직접 대입하기
s1 = 'name : {0}'.format('BlockDMask')
print(s1)

# 변수로 대입 하기
age = 55
s2 = 'age : {0}'.format(age)
print(s2)

# 이름으로 대입하기
s3 = 'number : {num}, gender : {gen}'.format(num=1234, gen='남')
print(s3)

# 인덱스를 입력하지 않으면?
s4 = 'name : {}, city : {}'.format('BlockDMask', 'seoul')
print(s4)

# 인덱스 순서가 바뀌면?
s5 = 'song1 : {1}, song2 : {0}'.format('nunu nana', 'ice cream')
print(s5)

# 인덱스를 중복해서 입력하면?
s6 = 'test1 : {0}, test2 : {1}, test3 : {0}'.format('인덱스0', '인덱스1')
print(s6)

# 나와라 중괄호
s7 = 'Format example. {{}}, {}'.format('test')
print(s7)

# 중괄호로 겹쳐진 인자값
s8 = 'This is value {{{0}}}'.format(1212)
print(s8)

# 왼쪽 정렬
s9 = 'this is {0:<10} | done {1:<5} |'.format('left', 'a')
print(s9)

# 오른쪽 정렬
s10 = 'this is {0:>10} | done {1:>5} |'.format('right', 'b')
print(s10)

# 가운데 정렬
s11 = 'this is {0:^10} | done {1:^5} |'.format('center', 'c')
print(s11)

# 문자열에 공백이 아닌 값 채우기
# 왼쪽 정렬
s12 = 'this is {0:-<10} | done {1:o<5} |'.format('left', 'a')
print(s12)

# 오른쪽 정렬
s13 = 'this is {0:+>10} | done {1:0>5} |'.format('right', 'b')
print(s13)

# 가운데 정렬
s14 = 'this is {0:.^10} | done {1:@^5} |'.format('center', 'c')
print(s14)

# 정수 N자리
s15 = '정수 3자리 : {0:03d}, {1:03d}'.format(12345, 12)
print(s15)

# 소수점 N자리
s16 = '아래 2자리 : {0:0.2f}, 아래 5자리 : {1:0.5f}'.format(123.1234567, 3.14)
print(s16)

# % 서식문자 -- %s, %d, %f, %o(8진수), %x(16진수), %%(문자%)
num = 50
s = 'my age %d' % num
print(s)

# 출력해야할 값이 두개 이상인 경우 () 를 이용합니다.
s1 = 'my name is %s. age : %d' % ('blockdmask', 100)
print(s1)

# --- f-string

# 정렬
s1 = 'left'
result1 = f'|{s1:<10}|'
print(result1)              # |left      |

s2 = 'mid'
result2 = f'|{s2:^10}|'
print(result2)              # |   mid    |

s3 = 'right'
result3 = f'|{s3:>10}|'
print(result3)              # |     right|

# 중괄호 출력
# -- f-string의 값과 중괄호를 같이 표현하려면 세개 입력해야 변수로 인식됨
num = 10
result = f'my age {{{num}}}, {{num}}'
print(result)


# 딕셔너리
d = {'name': 'BlockDMask', 'gender': 'man', 'age': 100}
result = f'my name {d["name"]}, gender {d["gender"]}, age {d["age"]}'
print(result)


# 리스트
n = [100, 200, 300]

print(f'list : {n[0]}, {n[1]}, {n[2]}')

for v in n:
    print(f'list with for : {v}')