{{/*
kubemark-ai Helm chart helpers
*/}}

{{- define "kubemark-ai.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "kubemark-ai.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "kubemark-ai.labels" -}}
app.kubernetes.io/name: {{ include "kubemark-ai.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app: kubemark-ai
{{- end }}

{{- define "kubemark-ai.gpusPerNode" -}}
{{- if .Values.gpu.gpusPerNode }}
{{- .Values.gpu.gpusPerNode }}
{{- else if or (eq .Values.gpu.type "gb200") (eq .Values.gpu.type "gb300") (eq .Values.gpu.type "h100x4") (eq .Values.gpu.type "a100x4") }}
{{- 4 }}
{{- else if or (eq .Values.gpu.type "h100") (eq .Values.gpu.type "a100") (eq .Values.gpu.type "b200") (eq .Values.gpu.type "b300") }}
{{- 8 }}
{{- else }}
{{- 1 }}
{{- end }}
{{- end }}

{{- define "kubemark-ai.numNodes" -}}
{{- if .Values.gpu.nodes }}
{{- .Values.gpu.nodes }}
{{- else }}
{{- div .Values.gpu.totalGPUs (include "kubemark-ai.gpusPerNode" . | int) }}
{{- end }}
{{- end }}
