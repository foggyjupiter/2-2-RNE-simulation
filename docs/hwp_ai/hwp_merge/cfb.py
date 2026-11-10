"""최소한의 복합 문서(CFB, v3, 512바이트 섹터) 쓰기. HWP 5 파일 컨테이너용.

entries: [(경로 튜플, bytes)] — 스트림만 주면 중간 저장소(storage)는 자동으로 만든다.
4096바이트 미만 스트림은 미니 스트림(64바이트 섹터)에 넣는다.
"""
import struct

SECT = 512
MINI = 64
CUTOFF = 4096
FREE, ENDCHAIN, FATSECT, DIFSECT = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFC
NOSTREAM = 0xFFFFFFFF


def _cmp_key(name):
    return (len(name), name.upper())


class _Node:
    def __init__(self, name, is_storage, data=b""):
        self.name, self.is_storage, self.data = name, is_storage, data
        self.children = {}
        self.sid = None
        self.left = self.right = self.child = NOSTREAM
        self.start = ENDCHAIN
        self.size = 0


def write(path, entries):
    root = _Node("Root Entry", True)
    for parts, data in entries:
        node = root
        for p in parts[:-1]:
            node = node.children.setdefault(p, _Node(p, True))
        node.children[parts[-1]] = _Node(parts[-1], False, data)

    # 디렉터리 순서 (너비 우선)
    order, queue = [], [root]
    while queue:
        n = queue.pop(0)
        n.sid = len(order)
        order.append(n)
        queue.extend(sorted(n.children.values(), key=lambda c: _cmp_key(c.name)))

    # 형제 트리: 정렬한 뒤 가운데를 뿌리로 하는 균형 이진 트리 (전부 검정)
    def build(nodes):
        if not nodes:
            return NOSTREAM
        m = len(nodes) // 2
        nodes[m].left = build(nodes[:m])
        nodes[m].right = build(nodes[m + 1:])
        return nodes[m].sid

    for n in order:
        if n.is_storage:
            n.child = build(sorted(n.children.values(), key=lambda c: _cmp_key(c.name)))

    # 미니 스트림
    mini_data = bytearray()
    minifat = []
    big = []
    for n in order:
        if n.is_storage:
            continue
        n.size = len(n.data)
        if n.size == 0:
            n.start = ENDCHAIN
        elif n.size < CUTOFF:
            first = len(minifat)
            k = (n.size + MINI - 1) // MINI
            for j in range(k):
                minifat.append(first + j + 1 if j < k - 1 else ENDCHAIN)
            n.start = first
            mini_data += n.data + b"\0" * (k * MINI - n.size)
        else:
            big.append(n)

    # 섹터 배치: [큰 스트림들][미니 스트림][미니FAT][디렉터리][FAT]
    sectors = []          # 각 원소: 512바이트
    fat = []

    def alloc(data):
        if not data:
            return ENDCHAIN
        k = (len(data) + SECT - 1) // SECT
        first = len(sectors)
        padded = data + b"\0" * (k * SECT - len(data))
        for j in range(k):
            sectors.append(padded[j * SECT:(j + 1) * SECT])
            fat.append(first + j + 1 if j < k - 1 else ENDCHAIN)
        return first

    for n in big:
        n.start = alloc(n.data)
    root.start = alloc(bytes(mini_data)) if mini_data else ENDCHAIN
    root.size = len(mini_data)
    mf = b"".join(struct.pack("<I", x) for x in minifat)
    if mf:
        mf += struct.pack("<I", FREE) * ((-len(mf) // 4) % (SECT // 4))
    minifat_start = alloc(mf) if mf else ENDCHAIN
    n_minifat = len(mf) // SECT

    # 디렉터리 항목
    d = bytearray()
    for n in order:
        name = n.name.encode("utf-16le") + b"\0\0"
        assert len(name) <= 64, n.name
        typ = 5 if n is root else (1 if n.is_storage else 2)
        e = name + b"\0" * (64 - len(name))
        e += struct.pack("<HBB", len(name), typ, 1)               # 1 = 검정
        e += struct.pack("<III", n.left, n.right, n.child if n.is_storage else NOSTREAM)
        e += b"\0" * 16 + struct.pack("<I", 0) + b"\0" * 16       # CLSID, state, 시간
        start = n.start if (not n.is_storage or n is root) else 0
        size = n.size if (not n.is_storage or n is root) else 0
        e += struct.pack("<III", start, size, 0)
        assert len(e) == 128
        d += e
    while len(d) % SECT:
        d += (b"\0" * 64 + struct.pack("<HBB", 0, 0, 0) + struct.pack("<III", NOSTREAM, NOSTREAM, NOSTREAM)
              + b"\0" * 36 + struct.pack("<III", 0, 0, 0))
    dir_start = alloc(bytes(d))

    # FAT 섹터 수 결정 (FAT 자신도 FAT에 들어간다)
    n_fat = 1
    while True:
        total = len(sectors) + n_fat
        if (total + (SECT // 4) - 1) // (SECT // 4) <= n_fat:
            break
        n_fat += 1
    assert n_fat <= 109, "DIFAT 섹터가 필요한 크기"
    fat_first = len(sectors)
    fat += [FATSECT] * n_fat
    fat += [FREE] * (n_fat * (SECT // 4) - len(fat))
    fb = b"".join(struct.pack("<I", x) for x in fat)
    for j in range(n_fat):
        sectors.append(fb[j * SECT:(j + 1) * SECT])

    hdr = bytearray()
    hdr += bytes.fromhex("D0CF11E0A1B11AE1") + b"\0" * 16
    hdr += struct.pack("<HHHHH", 0x3E, 3, 0xFFFE, 9, 6)
    hdr += b"\0" * 6
    hdr += struct.pack("<I", 0)              # 디렉터리 섹터 수 (v3는 0)
    hdr += struct.pack("<I", n_fat)
    hdr += struct.pack("<I", dir_start)
    hdr += struct.pack("<I", 0)              # 트랜잭션
    hdr += struct.pack("<I", CUTOFF)
    hdr += struct.pack("<II", minifat_start, n_minifat)
    hdr += struct.pack("<II", ENDCHAIN, 0)   # DIFAT 없음
    difat = [fat_first + j for j in range(n_fat)] + [FREE] * (109 - n_fat)
    hdr += b"".join(struct.pack("<I", x) for x in difat)
    assert len(hdr) == 512
    with open(path, "wb") as fh:
        fh.write(hdr)
        for s in sectors:
            fh.write(s)
