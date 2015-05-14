
import parse
import math
import json

def scalevec(a,s):
    (a1,a2,a3) = a
    return (s*a1,s*a2,s*a3)
def addvec(a,b):
    (a1,a2,a3) = a
    (b1,b2,b3) = b
    return (a1+b1,a2+b2,a3+b3)
def subvec(a,b):
    (a1,a2,a3) = a
    (b1,b2,b3) = b
    return (a1-b1,a2-b2,a3-b3)

def cross(a,b):
    (a1,a2,a3) = a
    (b1,b2,b3) = b
    return (a2*b3-a3*b2,-(a1*b3-a3*b1),a1*b2-a2*b1)

def normalize(a):
    (a1,a2,a3) = a
    n = math.sqrt(a1*a1+a2*a2+a3*a3)
    return (a1/n,a2/n,a3/n)

def readfile(filename):
    with open(filename,'r') as infile:
        fileFormat = None
        for line in infile:
            parsed = parse.parse("format {} {}",line)
            if parsed:
                fileFormat = parsed[0]

            # Get the number of vertices.
            parsed = parse.parse("element vertex {}",line)
            if parsed:
                vertexCount = int(parsed[0])

            # Get the number of faces.
            parsed = parse.parse("element face {}",line)
            if parsed:
                faceCount = int(parsed[0])

            if line == "end_header\n":
                break
        if fileFormat is None:
            print(fileFormat)
            return

        vertices = []
        for i in range(vertexCount):
            if fileFormat == "ascii":
                line = infile.readline()
                parsed = parse.parse("{} {} {}\n",line)
                x = float(parsed[0])
                y = float(parsed[1])
                z = float(parsed[2])
            elif fileFormat == "binary_big_endian":
                line = infile.read(12*2)
                (x,y,z) = struct.unpack(">f>f>f",line)
            vertices += [(x,y,z)]

        reuseCount = [0] * len(vertices)
        normals = [(0,0,0)] * len(vertices)
        faces = []
        for i in range(faceCount):
            if fileFormat == "ascii":
                line = infile.readline()
                parsed = parse.parse("3 {} {} {}\n",line)
                i1 = int(parsed[0])
                i2 = int(parsed[1])
                i3 = int(parsed[2])
            elif fileFormat == "binary_big_endian":
                line = infile.read(12*2)
                (x,y,z) = struct.unpack("xx>i>i>i",line)
            faces += [(i1,i2,i3)]

        for (i1,i2,i3) in faces:
            v1 = subvec(vertices[i2],vertices[i1])       
            v2 = subvec(vertices[i3],vertices[i1]) 
            n = normalize(cross(v1,v2))
            for j in [i1,i2,i3]:
                normals[j] = addvec(n,normals[j])
                reuseCount[j] += 1


        for j in range(len(vertices)):
            if reuseCount[j] > 1:
                normals[j] = scalevec(normals[j], 1 / reuseCount[j])

        m = {'indices':[],'vertices':[],'normals':[]}
        for n in faces:
            (x,y,z) = n
            m['indices'] += [[x,y,z]]
        for n in vertices:
            (x,y,z) = n
            m['vertices'] += [[x,y,z]]
        for n in normals:
            (x,y,z) = n
            m['normals'] += [[x,y,z]]

        print(json.JSONEncoder().encode(m))

readfile("Armadillo.ply")
