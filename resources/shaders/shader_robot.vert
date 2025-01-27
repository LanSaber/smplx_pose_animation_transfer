#version 330 core

const int MAX_JOINTS = 80;
const int MAX_WEIGHTS = 9;

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in mat3x3 jointIndex;
layout (location = 5) in mat3x3 jointWeight;
layout (location = 8) in vec3 aTangent;
layout (location = 9) in vec3 aBitangent;
layout (location = 10) in vec2 aTexCoord;

out vec2 TexCoord;
out vec3 fragPos;
out vec3 fragNormal;
out mat3 TBN;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform mat4 jointTransforms[MAX_JOINTS];

void main()
{

    vec4 totalLocalPos = vec4(0.0);
	vec4 totalNormal = vec4(0.0);

	vec4 totalTangent = vec4(0.0);
	vec4 totalBitangent = vec4(0.0);
	vec4 totalMicroNorm = vec4(0.0);

	for(int i = 0;i < MAX_WEIGHTS; i++){

        int index_i = i/3;
        int index_j = i%3;
		mat4 jointTransform = jointTransforms[int(jointIndex[index_i][index_j])];
		vec4 posePosition = jointTransform * vec4(aPos, 1.0);
		totalLocalPos += posePosition * jointWeight[index_i][index_j];

		vec4 worldNormal = jointTransform * vec4(aNormal, 0.0);
		totalNormal += worldNormal * jointWeight[index_i][index_j];

		vec4 worldTangent = jointTransform * vec4(aTangent, 0.0);
		totalTangent += worldTangent * jointWeight[index_i][index_j];

		vec4 worldBitangent = jointTransform * vec4(aBitangent, 0.0);
		totalBitangent += worldBitangent * jointWeight[index_i][index_j];
	}

	gl_Position = projection * view * model * totalLocalPos;

	// Transform fragment position to world space
    fragPos = vec3(model * totalLocalPos);

	// Calculate the normal in world space
	//	mat3 normalMatrix = mat3(transpose(inverse(model)));
	//	fragNormal = vec3(normalize(normalMatrix * vec3(totalMicroNorm))); // Transform and normalize the normal

	// calculate TBN
	vec3 T = normalize(vec3(model * totalTangent));
	vec3 B = normalize(vec3(model * totalBitangent));
	vec3 N = normalize(vec3(model * totalNormal));
	TBN = mat3(T, B, N);

	TexCoord = vec2(aTexCoord.x, aTexCoord.y);
}