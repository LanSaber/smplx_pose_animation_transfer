#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in vec3 Normal;
in vec3 FragPos;

uniform sampler2D diffuseMap;
uniform sampler2D specularMap;
uniform sampler2D normalMap;
uniform sampler2D glossinessMap;

uniform vec3 lightPos;
uniform vec3 viewPos;

void main() {
    vec3 color = texture(diffuseMap, TexCoords).rgb;

    vec3 normal = normalize(texture(normalMap, TexCoords).rgb * 2.0 - 1.0);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(lightDir, normal), 0.0);

    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);

    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0) * texture(specularMap, TexCoords).r;

    vec3 ambient = 0.1 * color;
    vec3 diffuse = diff * color;
    vec3 specular = spec * vec3(texture(glossinessMap, TexCoords).rgb);

    FragColor = vec4(ambient + diffuse + specular, 1.0);
}