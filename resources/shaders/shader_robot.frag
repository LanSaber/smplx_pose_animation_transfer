#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in vec3 fragPos;   // Fragment position from the vertex shader
in vec3 fragNormal;    // Normal in world space from the vertex shader
in mat3 TBN;

uniform sampler2D diffuseMap;
uniform sampler2D specularMap;
uniform sampler2D normalMap;

uniform vec3 lightPos;
uniform vec3 viewPos;

void main()
{

    // Normalize the normal
    // vec3 norm = normalize(fragNormal);

    // Sample the normal map
    vec3 norm = texture(normalMap, TexCoord).rgb;
    norm = norm * 2.0 - 1.0;  // Convert to [-1, 1] range
    norm = normalize(TBN*norm);

    // Compute light direction
    vec3 lightDir = normalize(lightPos - fragPos);


    // Diffuse lighting
    float diff = max(dot(norm, lightDir), 0.0);
	vec3 diffuseColor = diff * texture(diffuseMap, TexCoord).rgb;
    // Specular lighting

    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specularColor = texture(specularMap, TexCoord).rgb;
    specularColor = specularColor * spec;

    // Combine results
    vec3 result = diffuseColor + specularColor;

	// vec3 ambient = 0.1 * DiffuseColor
	FragColor = vec4(result, 1.0);


	//FragColor = vec4(0,0.2,0.3,0);
}