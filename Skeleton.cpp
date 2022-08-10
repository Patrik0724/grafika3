//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Benyovszki Patrik
// Neptun : IQ00CM
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

template<class T> struct Dnum {
	float f;
	T d;
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+ (Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator- (Dnum r) { return Dnum(f + r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (d * r.f - f * r.d) / r.f / r.f);
	}
};

template<class T> Dnum<T> Sin(Dnum<T> g) { return Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T> g) { return Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) { return Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d); }

typedef Dnum<vec2> Dnum2;

const int nTesselation = 100;

struct Camera {
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;

	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 45 * (float)M_PI / 180.0f;
		fp = 1;
		bp = 20;
	}

	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
													u.y, v.y, w.y, 0,
													u.z, v.z, w.z, 0,
													0, 0, 0, 1);
	}

	mat4 P() {
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
						0, 1 / tan(fov / 2), 0, 0,
						0, 0, -(fp + bp) / (bp - fp), -1,
						0, 0, -2 * fp * bp / (bp - fp), 0);
	}

	void animate(float dt) {
		vec3 d = wEye - wLookat;
		wEye = vec3(d.x * cos(dt) + d.y * sin(dt) + wLookat.x, d.y * cos(dt) - d.x * sin(dt) + wLookat.y, wEye.z);
	}
};

struct Material {
	vec3 kd, ks, ka;
	float shininess;
};

struct Light {
	vec3 Le, pos;
	
	void animate(mat4 transformation) {
		vec4 pos4 = vec4(pos.x, pos.y, pos.z, 1);
		vec4 newPos4 = pos4 * transformation;
		pos = vec3(newPos4.x, newPos4.y, newPos4.z);
	}
};

struct RenderState {
	mat4 MVP, M, V, P, Minv;
	Material* material;
	std::vector<Light> lights;
	vec3 wEye;
	vec4 t;
	float cosphimin;
};

class PhongShader : public GPUProgram {
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 Le, pos;
		};
		
		uniform mat4 MVP, M, Minv;
		uniform Light[2] lights;
		uniform vec3 wEye;
		uniform vec4 t;

		layout(location = 0) in vec3 vtxPos;
		layout(location = 1) in vec3 vtxNorm;

		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight[2];

		void main(){
			gl_Position = vec4(vtxPos, 1) * MVP;
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < 2; ++i){
				wLight[i] = lights[i].pos * wPos.w - wPos.xyz;
			}
			wView = wEye * wPos.w - wPos.xyz;
			wNormal = (Minv * vec4(vtxNorm, 0)).xyz; 
		}
	)";


	const char* fragmentSource = R"(
		#version 330
		precision highp float;
		
		struct Light {
			vec3 Le, pos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		const vec3 La = vec3(0.2f, 0.2f, 0.2f);

		uniform Material material;
		uniform Light[2] lights;
		uniform float cosphimin;
		uniform vec4 t;

		in vec3 wNormal;
		in vec3 wView;
		in vec3 wLight[2];

		out vec4 fragmentColor;
		
		void main(){
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);
			if(dot(N, V) < 0) N = -N;

			vec3 radiance = vec3(0, 0, 0);
			
			vec3 L = normalize(wLight[0]);
			vec3 H = normalize(L + V);
			float cost = max(dot(N, L), 0), cosd = max(dot(N, H), 0);
			vec3 LeIn = lights[0].Le / dot(wLight[0], wLight[0]);
			radiance += material.ka * La + (material.kd * cost + material.ks * pow(cosd, material.shininess)) * LeIn;

			L = normalize(wLight[1]);
			if(cosphimin < dot(-L, normalize(t.xyz)) || dot(wLight[1], wLight[1]) < 0.09){
				H = normalize(L + V);
				cost = max(dot(N, L), 0); cosd = max(dot(N, H), 0);
				LeIn = lights[1].Le / dot(wLight[1], wLight[1]);
				radiance += (material.kd * cost + material.ks * pow(cosd, material.shininess)) * LeIn;
			}		

			
			fragmentColor = vec4(radiance, 1);
		}
	)";

	void setUniformLight(Light light, std::string name) {
		setUniform(light.Le, name + ".Le");
		setUniform(light.pos, name + ".pos");
	}

	void setUniformMaterial(Material material, std::string name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }
	
	void Bind(RenderState state) {
		Use();
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");

		for (int i = 0; i < 2; ++i)
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		setUniform(state.cosphimin, "cosphimin");
		setUniform(state.t, "t");
	}
};

PhongShader* gpuProgram;

struct VertexData {
	vec3 position, normal;
};

class Geometry {
protected:
	unsigned int vao, vbo;
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

class ParamSurface : public Geometry {
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	void create() {
		nVtxPerStrip = (nTesselation + 1) * 2;
		nStrips = nTesselation;
		std::vector<VertexData> vtxData;
		for (int i = 0; i < nTesselation; ++i) {
			for (int j = 0; j <= nTesselation; ++j) {
				vtxData.push_back(GenVertexData((float)j / nTesselation, (float)i / nTesselation));
				vtxData.push_back(GenVertexData((float)j / nTesselation, (float)(i + 1) / nTesselation));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (int i = 0; i < nStrips; ++i) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};

class Sphere : public ParamSurface {
public:
	Sphere() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
		X = Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Cos(V);
	}
};

class Quad : public ParamSurface {
public:
	Quad() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		X = U; Y = V; Z = 0;
	}
};

class Cylinder : public ParamSurface {
public:
	Cylinder() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * M_PI;
		X = Cos(U); Y = Sin(U); Z = V;
	}
};

class Disc : public ParamSurface {
public:
	Disc() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * M_PI;
		X = Cos(U) * V; Y = Sin(U) * V; Z = 0;
	}
};

class Paraboloid : public ParamSurface {
public:
	Paraboloid() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2 * M_PI; 
		X = Cos(U) * V; Y = Sin(U) * V; Z = Pow(X, 2) + Pow(Y, 2);
	}
};

struct Floor {
	Material* material;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float phi;
	Floor(Material* _material, Geometry* _geometry, vec3 _scale, vec3 _translation, vec3 _rotationAxis, float _phi)
		: scale(_scale), translation(_translation), rotationAxis(_rotationAxis)
	{
		material = _material;
		geometry = _geometry;
		phi = _phi;
	}
	
	void Draw(RenderState &state) {
		state.M = ScaleMatrix(scale) * RotationMatrix(phi, rotationAxis) * TranslateMatrix(translation) * state.M;
		state.Minv = state.Minv * TranslateMatrix(-translation) * RotationMatrix(-phi, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		gpuProgram->Bind(state);
		geometry->Draw();
	}

};

struct LampElement {
	Material* material;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float phi;
	LampElement* next;
	LampElement(Material* _material, Geometry* _geometry, vec3 _scale, vec3 _translation, vec3 _rotationAxis, float _phi, LampElement* _next)
		: scale(_scale), translation(_translation), rotationAxis(_rotationAxis) 
	{
		material = _material;
		geometry = _geometry;
		phi = _phi;
		next = _next;
	}

	void Draw(RenderState &state) {
		state.M = RotationMatrix(phi, rotationAxis) * TranslateMatrix(translation) * state.M;
		state.Minv = state.Minv * TranslateMatrix(-translation) * RotationMatrix(-phi, rotationAxis);
		if (next != NULL)
			next->Draw(state);
		else {
			state.lights[1].animate(state.M);
			state.t = state.t * state.M;
		}
		state.M = ScaleMatrix(scale) * state.M;
		state.Minv = state.Minv * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		gpuProgram->Bind(state);
		geometry->Draw();
		state.M = TranslateMatrix(-translation) * RotationMatrix(-phi, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z)) * state.M;
		state.Minv = state.Minv * ScaleMatrix(scale) * RotationMatrix(phi, rotationAxis) * TranslateMatrix(translation);
	}

	void animate(float _phi) {
		phi = phi + _phi;
	}
};

struct Lamp {
	LampElement* first;

	float cosphimin;
	vec4 t;

	Lamp(LampElement* _first) {
		first = _first;
	}

	void animate(float phi, int i) {
		LampElement* start = first;
		for (int j = 0; j < i; ++j)
			start = start->next;
		start->animate(phi);
	}
};

class Scene {
	Lamp* l;
	Floor* floor;
	Camera camera;
	std::vector<Light> lights;
public:
	void Build() {
		Material* material = new Material;
		material->kd = vec3(0.2, 0.2, 0.5);
		material->ka = vec3(0.2 * M_PI, 0.2 * M_PI, 0.5 * M_PI);
		material->ks = vec3(2, 2, 2);
		material->shininess = 50;

		Material* material2 = new Material;
		material2->kd = vec3(0.5, 0.4, 0.3);
		material2->ka = vec3(0.5 * M_PI, 0.4 * M_PI, 0.3 * M_PI);
		material2->ks = vec3(2, 2, 2);
		material2->shininess = 50;

		LampElement* lampshade = new LampElement(material, new Paraboloid(), vec3(0.3, 0.3, 0.3), vec3(0, 0, 0.045), vec3(0, 0, 1), 0, NULL);

		LampElement* sphere3 = new LampElement(material, new Sphere, vec3(0.05, 0.05, 0.05), vec3(0, 0, 0.63), vec3(2, 2, 3), -1.5, lampshade);

		LampElement* stick2 = new LampElement(material, new Cylinder, vec3(0.02, 0.02, 0.6), vec3(0, 0, 0.03), vec3(1, 0, 0), 0, sphere3);

		LampElement* sphere2 = new LampElement(material, new Sphere, vec3(0.05, 0.05, 0.05), vec3(0, 0, 1.23), vec3(1, 0, 1), M_PI, stick2);

		LampElement* stick1 = new LampElement(material, new Cylinder, vec3(0.02, 0.02, 1.2), vec3(0, 0, 0.03), vec3(0, 1, 0), 0, sphere2);

		LampElement* sphere1 = new LampElement(material, new Sphere(), vec3(0.05, 0.05, 0.05), vec3(0, 0, 0.03), vec3(0, 1, 5), M_PI, stick1);

		LampElement* foottop = new LampElement(material, new Disc(), vec3(0.5, 0.4, 1), vec3(0, 0, 0.1), vec3(0, 0, 1), 0, sphere1);

		LampElement* foot = new LampElement(material, new Cylinder(), vec3(0.5, 0.4, 0.1), vec3(0, 0, 0), vec3(0, 0, 1), 0, foottop);

		floor = new Floor(material2, new Quad(), vec3(-100, -100, 1), vec3(10, 10, 0), vec3(0, 0, 1), 0);

		
		l = new Lamp(foot);
		

		camera.wEye = vec3(2, 2, 4);
		camera.wLookat = vec3(0, 0, 1);
		camera.wVup = vec3(0, 0, 1);

		lights.resize(2);
		lights[0].Le = vec3(8, 8, 8);
		lights[0].pos = vec3(0, 2, 5);

		lights[1].Le = vec3(2, 2, 2);
		lights[1].pos = vec3(0, 0, 0.3);
	}

	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		state.M = ScaleMatrix(vec3(1, 1, 1));
		state.Minv = ScaleMatrix(vec3(1, 1, 1));
		state.cosphimin = 0.76;
		state.t = vec4(0, 0, 1, 0);
		state.M = ScaleMatrix(vec3(1, 1, 1));
		state.Minv = ScaleMatrix(vec3(1, 1, 1));
		l->first->Draw(state);
		floor->Draw(state);
	}

	void animate(float dt) {
		camera.animate(0.5 * dt);

		l->animate(dt, 2);
		l->animate(dt, 4);
		l->animate(dt, 6);
	}
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	gpuProgram = new PhongShader();
	scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer
	scene.Render();
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	scene.animate(0.01f);
	glutPostRedisplay();
}
