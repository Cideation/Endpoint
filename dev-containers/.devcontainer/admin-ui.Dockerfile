FROM node:18-alpine
WORKDIR /app
COPY ./admin-ui /app
RUN npm install
CMD ["npm", "run", "dev"] 