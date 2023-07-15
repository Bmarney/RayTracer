import math as math
import sys;
import numpy as np
from numpy.linalg import inv

class Vector:
    #create initializer
    def __init__(self, xval = 0.0, yval = 0.0, zval = 0.0):
        self.x = xval
        self.y = yval
        self.z = zval

    #overload add operator for vector class
    def __add__(self, adder):
        #asserts isinstance(adder, Vector)
        newx = self.x + adder.x
        newy = self.y + adder.y
        newz = self.z + adder.z
        added = Vector(newx, newy, newz)
        return added
  
    #overload subtract operator for vector class
    def __sub__(self, subber):
        #assert isinstance(subber, Vector)
        newx = self.x - subber.x
        newy = self.y - subber.y
        newz = self.z - subber.z
        subbed = Vector(newx, newy, newz)
        return subbed

    #overload scalar multiplication operator for vector class
    def __mul__(self, multi):
        assert not isinstance(multi, Vector) #this would be cross product otherwise
        newx = float(self.x * multi)
        newy = float(self.y * multi)
        newz = float(self.z * multi)
        multiplied = Vector(newx, newy, newz)
        return multiplied

    #overload the rmul just in case it were to be used
    def __rmul__(self, multi):
        assert not isinstance(multi, Vector) #this would be cross product otherwise
        return self.__mul__(multi)

    #overload division operator for vector class
    def __truediv__(self, dividend):
        assert not isinstance(dividend, Vector) # just no...
        divx = self.x / dividend
        divy = self.y / dividend
        divz = self.z / dividend
        dividended = Vector(divx, divy, divz) #this name made me smile
        return dividended 

    #define string method for vector class
    def __str__(self):
        return "({}, {}, {})".format (self.x, self.y, self.z)

    #create dot product generator method
    def dot(self, prod):
        assert isinstance(prod, Vector)
        dotx = self.x * prod.x
        doty = self.y * prod.y
        dotz = self.z * prod.z
        dotted = float(dotx + doty + dotz)
        return dotted
  
    #create cross product generator method
    def cross(self, prod):
        assert isinstance(prod, Vector)
        crossx = (self.y * prod.z) - (self.z * prod.y)
        crossy = (self.z * prod.x) - (self.x * prod.z)
        crossz = (self.x * prod.y) - (self.y * prod.z)
        crossed = Vector(crossx, crossy, crossz)
        return crossed

    #create magnitude generator method
    def mag(self):
        magnitude = math.sqrt(self.dot(self))
        return magnitude

    #create normalizer method
    def normalize(self):
        norm = self/self.mag()
        return norm

#=============================================================================================

class Image:
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pixels = [[None for _ in range(width)]for _ in range(height)]


    def setPixel(self, x, y, color):
        self.pixels[y][x] = color

  
    def gen_ppm(self, img_file):
        def writeByte(color):
            col = round(min(color*255, 255)) # doesnt allow for anything above max color value
            col = max(0, col) # no negative color values
            return col

        img_file.write("P3 {} {} \n255\n".format(self.width, self.height))
    
        for rows in reversed(self.pixels):
            for color in rows:
                img_file.write("{} {} {} ".format(writeByte(color.x), writeByte(color.y), writeByte(color.z)))
            img_file.write("\n")

#========================================================================================================================

class Color(Vector):
     #using vector notation allows for storage of color values    
     #this just helped me distinguish colors from vectors
    pass
    
#===============================================================================================================================

class Point(Vector):
    #using vector's notation allows for storage and use of points
    #to be honest I kinda used this randomly I planned to make destinctions between points and vecotrs but it just never mattered really so I 
    #kinda used this just for some spice and could have removed it
    pass

#===============================================================================================================================

class Sphere:
    #the only shapes we are using for this ray tracer
    #(self, name, posx, posy, posz, sclx, scly, sclz, r, g, b, K_a, K_d, K_s, K_r, n)
    
    def __init__(self, position, scaling, material):
        self.position = Vector(position.x, position.y, position.z)
        self.scaling = Vector(scaling.x, scaling.y, scaling.z)
        self.material = material
        self.radius = 1
        self.model_matrix = np.array([
        [scaling.x, 0, 0, position.x],
        [0, scaling.y, 0, position.y],
        [0, 0, scaling.z, position.z],
        [0, 0, 0, 1]
        ])
        
        self.matrix_inv = inv(self.model_matrix)
        self.inverse_transpose = np.transpose(self.matrix_inv)


    def intersects(self, ray):
        #check to see if rays intersect this instance of a sphere and returns the distance(s) or NONE if there isn't one
        
        rayMatMulDir = np.array([ray.direction.x, ray.direction.y, ray.direction.z, 0])
        direction_raytemp = np.matmul(self.matrix_inv, rayMatMulDir)
        rayMatMulPoint = np.array([ray.origin.x, ray.origin.y, ray.origin.z, 1])
        sphere_to_raytemp = np.matmul(self.matrix_inv, rayMatMulPoint)
        newRay = Ray(Vector(sphere_to_raytemp[0], sphere_to_raytemp[1], sphere_to_raytemp[2]),Vector(direction_raytemp[0],direction_raytemp[1],direction_raytemp[2])) 
       
           
        # |c|^2*t^2 + 2*(S Dot c)t +|S^2| -1 = 0

        #multiply c and S by M^-1 for transformations
        
        #INTERSECT THE INVERSE TRANSFORMED RAY WITH THE UNTRANSFORMED PRIMITIVE

        
        a = (newRay.direction.dot(newRay.direction) * newRay.direction.dot(newRay.direction))
       
        
        b = 2 * newRay.origin.dot(newRay.direction)
        c = newRay.origin.dot(newRay.origin) -1 
        solution = (b*b) - (4*a*c)


        if (solution >=0):
            #breakpoint()
            dist = (-(b) - math.sqrt(solution)) / (2*a) 
            if (dist > 0):
                return dist
        return None
  
    def normal(self, surface_point):
        # find the normal of a surface point and return it

        surface_mult = np.array([surface_point.x, surface_point.y, surface_point.z, 1])
        temp_norm = np.matmul(self.inverse_transpose, surface_mult)
        normal = Vector(temp_norm[0], temp_norm[1], temp_norm[2])
        
        return (surface_point - self.position).normalize()
        
  
 #===============================================================================================================================
  
class Ray:
    # a normalized vector direction with a point origin
    def __init__(self, origin, direction):
        self.origin = Vector(origin.x, origin.y, origin.z)
        self.direction = direction.normalize()

#===============================================================================================================================

class Scene:
    #store all the data needed for raytracing
    
    
    def __init__(self, eye, objects, lights, width, height, numSpheres, numLights, ambientI, background, left, right, top, bottom, near):
        self.eye = eye
        self.width = width
        self.height = height
        self.objects = objects
        self.lights = lights
        self.numSpheres = numSpheres
        self.numLights = numLights
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.near = near
        self.ambientI = ambientI
        self.background = background
        

#===============================================================================================================================

class RayTracer:
    #backbone, heavy lifting and the source of all my pain
    #used for raytracing the scene
    max_depth = 3 # dont spawn more than 3 bounces as per specs
    min_displacement = 0.0001 #also from specs

    def trace(self, scene): 
        width = scene.width
        height = scene.height
        left = scene.left
        right = scene.right
        top = scene.top
        bottom = scene.bottom
        near = scene.near
        
        
        x_0 = left
        x_1 = right
        x_step = (x_1 - x_0) / (width - 1) 
        y_0 = bottom
        y_1 = top
        y_step = (y_1 - y_0) / (height - 1) 

        eye = scene.eye 
        pixels = Image(width, height) #creates a virtual screen with a color value available for each pixel

        for j in range(height): 
            y = y_0 + j*y_step
            for i in range(width):
                x = x_0 + i*x_step
                ray = Ray(eye, Point(x,y,((-1)*(near))) - eye)
                pixels.setPixel(i,j,self.ray_trace(ray, scene))
            print("{:3.0f}%".format((float(j)/float(height) * 100)), end="\r")
        return pixels

    def ray_trace(self, ray, scene, depth = 0):
        # if depth > MAX_DEPTH return BACKGROUND COLOR
        color = scene.background 
        #find nearest intersection if any

        #distance_hit is to P (point_hit)
        distance_hit, object_hit = self.find_nearest(ray, scene) # object_hit is an extra copy that can be overwritten in recursion
        if object_hit is None:
            return color
        
        hit_position = ray.origin + ray.direction * distance_hit
        #print(hit_position)
        color = Color(0,0,0) #dont add anything if reflected rays miss
        hit_normal = object_hit.normal(hit_position)
        color += self.color_at(object_hit, hit_position, hit_normal, scene)
        if depth < self.max_depth:
            new_ray_pos = hit_position + hit_normal * self.min_displacement # to avoid float point errors
            new_ray_dir = (ray.direction - 2 * ray.direction.dot(hit_normal) * hit_normal)
            new_ray = Ray(new_ray_pos, new_ray_dir)
            
            color += (self.ray_trace(new_ray, scene, depth + 1) * object_hit.material.reflection) # Attenuate the reflected ray by the reflection coefficient

        return color
        
    #find the nearest intersection
    def find_nearest(self, ray, scene):
        distance_min = None
        object_hit = None
        #for i in range(scene.numSpheres):
        #print(scene.objects)
        for obj in scene.objects:
            #breakpoint()
            #print(obj)
            dist = obj.intersects(ray)
            if dist is not None and (object_hit is None or dist < distance_min):
                distance_min = dist
                object_hit = obj
        return(distance_min, object_hit)

    #find the color of an object
    def color_at(self, object_hit, hit_position, normal, scene):
        material = object_hit.material
        object_color = material.color_at(hit_position)
        ambient_I = scene.ambientI
        
        to_cam = scene.eye - hit_position
        color = Color(0,0,0)
        color.x = ambient_I.x * object_color.x * material.ambient
        color.y = ambient_I.x * object_color.y * material.ambient
        color.z = ambient_I.x * object_color.z * material.ambient
        #light calculations
        for light in scene.lights:
            to_light = Ray(hit_position * self.min_displacement, light.position - hit_position *self.min_displacement)

            #SHADOWS NOT WORKING PROPERLY
            #if self.find_nearest(to_light, scene) is None:

            #diffuse shading
            color.x += material.diffuse * light.color.x * max(normal.dot(to_light.direction),0) * object_color.x
            color.y += material.diffuse * light.color.y * max(normal.dot(to_light.direction),0) * object_color.y
            color.z += material.diffuse * light.color.z * max(normal.dot(to_light.direction),0) * object_color.z

            #specular shading

            #reflected_ray = 2*(normal.dot(to_light.direction))*normal - to_light.direction
            #color += (light.color * material.specular * ((max(reflected_ray.dot(to_cam),0)** material.n)))
    

            reflector = (to_light.direction + to_cam).normalize()
            color += (light.color * material.specular * (max(normal.dot(reflector),0)** material.n))
        return color

#===============================================================================================================================

class Light:
    # LIGHT <name> <pos x> <pos y> <pos z> <Ir> <Ig> <Ib>
    
    def __init__(self, position, color=Color(1.0,1.0,1.0)): 
        self.position = position
        self.color = color

#===============================================================================================================================

class Material:
    #<Ka> <Kd> <Ks> <Kr> <n> from Spheres
    def __init__(self, color=Color(1.0,1.0,1.0), ambient = 0.05, diffuse = 1.0, specular = 1.0, reflection = 0.5, n = 5):
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.reflection = reflection
        self.n = n

    def color_at(self, position):
       return self.color

#===============================================================================================================================

if __name__ == "__main__":
    
    possibleSpheres = 15 #rows
    sphereData = 15 #columns
    spheres = [[None for _ in range(possibleSpheres)]for _ in range(sphereData)]
    possibleLights = 9 #rows
    lightData = 7 #columns
    lights = [[None for _ in range(possibleLights)]for _ in range(lightData)]
    back = [None for _ in range(3)]
    resolution = [None for _ in range(2)]
    ambient = [None for _ in range(3)]
    near = 0
    left = 0
    bottom = 0
    top = 0 
    right = 0
    sphereCount = 0
    lightCount = 0
    output = "test.ppm"
    #print(sys.argv[1]);
    f = open(sys.argv[1], "r")
    indexCount = 0
    for line in f:
        parseArr = line.split()
        if len(parseArr) == 0:
            continue
        for elements in parseArr:
            if elements == "NEAR":
                    #print(element)
                    near = (parseArr[1])
                    #print(near)
            elif elements == "LEFT":
                    #print(element)
                    left = (parseArr[1])
                    #print(left)
            elif elements == "RIGHT":
                    #print(element)
                    right = (parseArr[1])
                    #print(right)
            elif elements == "BOTTOM":
                    #print(element)
                    bottom = (parseArr[1])
                    #print(bottom)
            elif elements == "TOP":
                    #print(element)
                    top = (parseArr[1])
                    #print(top)
            elif elements == "RES":
                    resolution[0] = parseArr[1]
                    resolution[1] = parseArr[2]
                    #print(resolution[0])
                    #print(resolution[1])
            elif elements == "SPHERE":
                    for i in range(sphereData):
                        if i >0:
                            spheres[sphereCount][i] = float(parseArr[i+1])
                            #print(spheres[sphereCount][i])
                    sphereCount +=1
            elif elements == "LIGHT":
                    for j in range(lightData):
                        if j >0:                    
                            lights[lightCount][j] = float(parseArr[j+1])
                            #print(lights[lightCount][j])
                    lightCount += 1
            elif elements == "BACK":
                    for k in range(3):
                        back[k] = parseArr[k+1]
                        #print(back[k])
            elif elements == "AMBIENT":
                    for k in range(3):
                        ambient[k] = parseArr[k+1]
                        #print(ambient[k])
            elif elements == "OUTPUT":
                    output = parseArr[1]
                    #print(output)

    #initialize the arrays/lists
    sphere_Position = [None for _ in range(sphereCount)]
    sphere_Color = [None for _ in range(sphereCount)]
    sphere_Scale = [None for _ in range(sphereCount)]
    sphere_Materials = [None for _ in range(sphereCount)]
    objects = [None for _ in range(sphereCount)]
    light_Position = [None for _ in range(lightCount)]
    light_Color = [None for _ in range(lightCount)]
    lights_Pass = [None for _ in range(lightCount)]

    
    #set up all the sphere data
    for i in range(sphereCount):
        sphere_Position[i] = Point(spheres[i][1], spheres[i][2], spheres[i][3])
        sphere_Scale[i] = Point(spheres[i][4], spheres[i][5], spheres[i][6])
        sphere_Color[i] = Color(spheres[i][7], spheres[i][8], spheres[i][9])
        sphere_Materials[i] = Material(sphere_Color[i],spheres[i][10], spheres[i][11], spheres[i][12], spheres[i][13], spheres[i][14])
        #objects[i] = Sphere(sphere_Position[i], 1, sphere_Materials[i])
        #print(objects[i].position)
        #objects = Sphere(sphere_Position, 1, sphere_Materials)

    #set up the object data
    for i in range(sphereCount):
        objects[i] = Sphere(sphere_Position[i], sphere_Scale[i], sphere_Materials[i])
        #print(objects[i].position, objects[i].material.color)
    
    #set up the light data
    for i in range(lightCount):
        light_Position[i] = Point(lights[i][1], lights[i][2], lights[i][3])
        light_Color[i] = Color(lights[i][4], lights[i][5], lights[i][6])

    #set up the lights_pass data
    for i in range(lightCount):
        lights_Pass[i] = Light(light_Position[i], light_Color[i])
        

    background_color = Color(float(back[0]), float(back[1]), float(back[2]))
    ambient_light = Color(float(ambient[0]), float(ambient[1]), float(ambient[2]))

    width = int(resolution[0])
    height = int(resolution[1])
    eye = Vector(0,0,0)

    #store all the variables that will be needed later
    scene = Scene(eye, objects, lights_Pass, width, height, sphereCount, lightCount, ambient_light, background_color, float(left), float(right), float(top), float(bottom), float(near))

    #do the raytracing!!
    raytracing = RayTracer()
    image = raytracing.trace(scene)
    
    #output PPM file
    with open(output, "w") as img_file:
       image.gen_ppm(img_file)
